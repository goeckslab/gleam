import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ImageLearner")


def split_data_0_2(
    df: pd.DataFrame,
    split_column: str,
    validation_size: float = 0.1,
    random_state: int = 42,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Given a DataFrame whose split_column only contains {0,2}, re-assign a portion of the 0s to become 1s (validation)."""
    out = df.copy()
    out[split_column] = pd.to_numeric(out[split_column], errors="coerce").astype(int)

    idx_train = out.index[out[split_column] == 0].tolist()

    if not idx_train:
        logger.info("No rows with split=0; nothing to do.")
        return out
    stratify_arr = None
    if label_column and label_column in out.columns:
        label_counts = out.loc[idx_train, label_column].value_counts()
        if label_counts.size > 1:
            # Force stratify even with fewer samples - adjust validation_size if needed
            min_samples_per_class = label_counts.min()
            if min_samples_per_class * validation_size < 1:
                # Adjust validation_size to ensure at least 1 sample per class, but do not exceed original validation_size
                adjusted_validation_size = min(
                    validation_size, 1.0 / min_samples_per_class
                )
                if adjusted_validation_size != validation_size:
                    validation_size = adjusted_validation_size
                    logger.info(
                        f"Adjusted validation_size to {validation_size:.3f} to ensure at least one sample per class in validation"
                    )
            stratify_arr = out.loc[idx_train, label_column]
            logger.info("Using stratified split for validation set")
        else:
            logger.warning("Only one label class found; cannot stratify")
    if validation_size <= 0:
        logger.info("validation_size <= 0; keeping all as train.")
        return out
    if validation_size >= 1:
        logger.info("validation_size >= 1; moving all train → validation.")
        out.loc[idx_train, split_column] = 1
        return out
    # Always try stratified split first
    try:
        train_idx, val_idx = train_test_split(
            idx_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify_arr,
        )
        logger.info("Successfully applied stratified split")
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}); falling back to random split.")
        train_idx, val_idx = train_test_split(
            idx_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=None,
        )
    out.loc[train_idx, split_column] = 0
    out.loc[val_idx, split_column] = 1
    out[split_column] = out[split_column].astype(int)
    return out


def create_stratified_random_split(
    df: pd.DataFrame,
    split_column: str,
    split_probabilities: list = [0.7, 0.1, 0.2],
    random_state: int = 42,
    label_column: Optional[str] = None,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Create a stratified random split when no split column exists."""
    out = df.copy()

    # initialize split column
    out[split_column] = 0

    if group_column and group_column not in out.columns:
        logger.warning(
            "Group column '%s' not found in data; proceeding without group-aware split.",
            group_column,
        )
        group_column = None

    def _allocate_split_counts(n_total: int, probs: list) -> list:
        """Allocate exact split counts using largest remainder rounding.

        Ensures at least one sample per active split *after* proportional allocation,
        by moving samples from other splits when possible. Tie-breaking for leftovers
        and corrections prioritizes: train (0), test (2), validation (1).
        """
        if n_total <= 0:
            return [0 for _ in probs]

        counts = [0 for _ in probs]
        active = [i for i, p in enumerate(probs) if p > 0]
        if not active:
            return counts

        # If there are fewer samples than active splits, fill in order and return.
        if n_total < len(active):
            priority_order = [0, 2, 1]
            ordered = [i for i in priority_order if i in active] + [
                i for i in active if i not in priority_order
            ]
            remaining = n_total
            for idx in ordered:
                if remaining <= 0:
                    break
                counts[idx] = 1
                remaining -= 1
            return counts

        probs_arr = np.array(probs, dtype=float)
        total_prob = probs_arr.sum()
        if total_prob <= 0:
            return counts
        probs_arr = probs_arr / total_prob
        raw = n_total * probs_arr
        floors = np.floor(raw).astype(int)
        counts = floors.tolist()

        leftover = n_total - int(floors.sum())
        if leftover > 0:
            frac = raw - floors
            priority_order = [0, 2, 1]
            order = sorted(
                active,
                key=lambda i: (-frac[i], priority_order.index(i) if i in priority_order else 999),
            )
            for i in range(leftover):
                counts[order[i % len(order)]] += 1

        # Ensure at least one per active split by moving from other splits.
        missing = [i for i in active if counts[i] == 0]
        if missing:
            priority_order = [0, 2, 1]
            missing_ordered = [i for i in priority_order if i in missing] + [
                i for i in missing if i not in priority_order
            ]
            for idx in missing_ordered:
                donors = [i for i in active if counts[i] > 1 and i != idx]
                if not donors:
                    break
                # Prefer taking from lower-priority splits first (val -> test -> train)
                donor_priority = [1, 2, 0]
                donors_sorted = sorted(
                    donors,
                    key=lambda i: (
                        -counts[i],
                        donor_priority.index(i) if i in donor_priority else 999,
                    ),
                )
                donor = donors_sorted[0]
                counts[donor] -= 1
                counts[idx] += 1

        return counts

    def _choose_split(counts: list, targets: list, active: list) -> int:
        remaining = [targets[i] - counts[i] for i in range(len(targets))]
        best = max(active, key=lambda i: (remaining[i], -counts[i], -targets[i]))
        if remaining[best] <= 0:
            best = min(active, key=lambda i: counts[i])
        return best

    if not label_column or label_column not in out.columns:
        logger.warning(
            "No label column found; using random split without stratification"
        )
        # fall back to random assignment (group-aware if requested)
        indices = out.index.tolist()
        rng = np.random.RandomState(random_state)

        if group_column:
            group_series = out[group_column].astype(object)
            missing_mask = group_series.isna()
            if missing_mask.any():
                group_series = group_series.copy()
                group_series.loc[missing_mask] = [
                    f"__missing__{idx}" for idx in group_series.index[missing_mask]
                ]
            group_to_indices = {}
            for idx, group_id in group_series.items():
                group_to_indices.setdefault(group_id, []).append(idx)
            group_ids = sorted(group_to_indices.keys(), key=lambda x: str(x))
            rng.shuffle(group_ids)

            targets = _allocate_split_counts(len(indices), split_probabilities)
            counts = [0 for _ in split_probabilities]
            active = [i for i, p in enumerate(split_probabilities) if p > 0]
            train_idx = []
            val_idx = []
            test_idx = []

            for group_id in group_ids:
                size = len(group_to_indices[group_id])
                split_idx = _choose_split(counts, targets, active)
                counts[split_idx] += size
                if split_idx == 0:
                    train_idx.extend(group_to_indices[group_id])
                elif split_idx == 1:
                    val_idx.extend(group_to_indices[group_id])
                else:
                    test_idx.extend(group_to_indices[group_id])

            out.loc[train_idx, split_column] = 0
            out.loc[val_idx, split_column] = 1
            out.loc[test_idx, split_column] = 2
            return out.astype({split_column: int})

        rng.shuffle(indices)

        targets = _allocate_split_counts(len(indices), split_probabilities)
        n_train, n_val, n_test = targets

        out.loc[indices[:n_train], split_column] = 0
        out.loc[indices[n_train:n_train + n_val], split_column] = 1
        out.loc[indices[n_train + n_val:n_train + n_val + n_test], split_column] = 2

        return out.astype({split_column: int})

    # check if stratification is possible
    label_counts = out[label_column].value_counts()
    min_samples_per_class = label_counts.min()

    # ensure we have enough samples for stratification:
    # Each class must have at least as many samples as the number of nonzero splits,
    # so that each split can receive at least one sample per class.
    active_splits = [p for p in split_probabilities if p > 0]
    min_samples_required = len(active_splits)
    if min_samples_per_class < min_samples_required:
        logger.warning(
            f"Insufficient samples per class for stratification (min: {min_samples_per_class}, required: {min_samples_required}); using random split"
        )
        # fall back to simple random assignment (group-aware if requested)
        indices = out.index.tolist()
        rng = np.random.RandomState(random_state)

        if group_column:
            group_series = out[group_column].astype(object)
            missing_mask = group_series.isna()
            if missing_mask.any():
                group_series = group_series.copy()
                group_series.loc[missing_mask] = [
                    f"__missing__{idx}" for idx in group_series.index[missing_mask]
                ]
            group_to_indices = {}
            for idx, group_id in group_series.items():
                group_to_indices.setdefault(group_id, []).append(idx)
            group_ids = sorted(group_to_indices.keys(), key=lambda x: str(x))
            rng.shuffle(group_ids)

            targets = _allocate_split_counts(len(indices), split_probabilities)
            counts = [0 for _ in split_probabilities]
            active = [i for i, p in enumerate(split_probabilities) if p > 0]
            train_idx = []
            val_idx = []
            test_idx = []

            for group_id in group_ids:
                size = len(group_to_indices[group_id])
                split_idx = _choose_split(counts, targets, active)
                counts[split_idx] += size
                if split_idx == 0:
                    train_idx.extend(group_to_indices[group_id])
                elif split_idx == 1:
                    val_idx.extend(group_to_indices[group_id])
                else:
                    test_idx.extend(group_to_indices[group_id])

            out.loc[train_idx, split_column] = 0
            out.loc[val_idx, split_column] = 1
            out.loc[test_idx, split_column] = 2
            return out.astype({split_column: int})

        rng.shuffle(indices)

        targets = _allocate_split_counts(len(indices), split_probabilities)
        n_train, n_val, n_test = targets

        out.loc[indices[:n_train], split_column] = 0
        out.loc[indices[n_train:n_train + n_val], split_column] = 1
        out.loc[indices[n_train + n_val:n_train + n_val + n_test], split_column] = 2

        return out.astype({split_column: int})

    if group_column:
        logger.info(
            "Using stratified random split for train/validation/test sets (grouped by '%s')",
            group_column,
        )
        rng = np.random.RandomState(random_state)

        group_series = out[group_column].astype(object)
        missing_mask = group_series.isna()
        if missing_mask.any():
            group_series = group_series.copy()
            group_series.loc[missing_mask] = [
                f"__missing__{idx}" for idx in group_series.index[missing_mask]
            ]

        group_to_indices = {}
        for idx, group_id in group_series.items():
            group_to_indices.setdefault(group_id, []).append(idx)

        group_ids = sorted(group_to_indices.keys(), key=lambda x: str(x))
        group_labels = {}
        mixed_groups = []
        label_series = out[label_column]

        for group_id in group_ids:
            labels = label_series.loc[group_to_indices[group_id]].dropna().unique()
            if len(labels) == 1:
                group_labels[group_id] = labels[0]
            elif len(labels) == 0:
                group_labels[group_id] = None
            else:
                mode_vals = label_series.loc[group_to_indices[group_id]].mode(dropna=True)
                group_labels[group_id] = mode_vals.iloc[0] if not mode_vals.empty else labels[0]
                mixed_groups.append(group_id)

        if mixed_groups:
            logger.warning(
                "Detected %d groups with multiple labels; using the most common label per group for stratification.",
                len(mixed_groups),
            )

        train_idx = []
        val_idx = []
        test_idx = []
        active = [i for i, p in enumerate(split_probabilities) if p > 0]

        for label_value in sorted(label_counts.index.tolist(), key=lambda x: str(x)):
            label_groups = [g for g in group_ids if group_labels.get(g) == label_value]
            if not label_groups:
                continue
            rng.shuffle(label_groups)
            label_total = sum(len(group_to_indices[g]) for g in label_groups)
            targets = _allocate_split_counts(label_total, split_probabilities)
            counts = [0 for _ in split_probabilities]

            for group_id in label_groups:
                size = len(group_to_indices[group_id])
                split_idx = _choose_split(counts, targets, active)
                counts[split_idx] += size
                if split_idx == 0:
                    train_idx.extend(group_to_indices[group_id])
                elif split_idx == 1:
                    val_idx.extend(group_to_indices[group_id])
                else:
                    test_idx.extend(group_to_indices[group_id])

        # Assign groups without a label (or missing labels) using overall targets.
        unlabeled_groups = [g for g in group_ids if group_labels.get(g) is None]
        if unlabeled_groups:
            rng.shuffle(unlabeled_groups)
            total_unlabeled = sum(len(group_to_indices[g]) for g in unlabeled_groups)
            targets = _allocate_split_counts(total_unlabeled, split_probabilities)
            counts = [0 for _ in split_probabilities]
            for group_id in unlabeled_groups:
                size = len(group_to_indices[group_id])
                split_idx = _choose_split(counts, targets, active)
                counts[split_idx] += size
                if split_idx == 0:
                    train_idx.extend(group_to_indices[group_id])
                elif split_idx == 1:
                    val_idx.extend(group_to_indices[group_id])
                else:
                    test_idx.extend(group_to_indices[group_id])

        out.loc[train_idx, split_column] = 0
        out.loc[val_idx, split_column] = 1
        out.loc[test_idx, split_column] = 2

        logger.info("Successfully applied stratified random split")
        logger.info(
            f"Split counts: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )
        return out.astype({split_column: int})

    logger.info("Using stratified random split for train/validation/test sets (per-class allocation)")

    rng = np.random.RandomState(random_state)
    label_values = sorted(label_counts.index.tolist(), key=lambda x: str(x))
    train_idx = []
    val_idx = []
    test_idx = []

    for label_value in label_values:
        label_indices = out.index[out[label_column] == label_value].tolist()
        rng.shuffle(label_indices)
        n_train, n_val, n_test = _allocate_split_counts(len(label_indices), split_probabilities)
        train_idx.extend(label_indices[:n_train])
        val_idx.extend(label_indices[n_train:n_train + n_val])
        test_idx.extend(label_indices[n_train + n_val:n_train + n_val + n_test])

    # assign split values
    out.loc[train_idx, split_column] = 0
    out.loc[val_idx, split_column] = 1
    out.loc[test_idx, split_column] = 2

    logger.info("Successfully applied stratified random split")
    logger.info(
        f"Split counts: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )
    return out.astype({split_column: int})


class SplitProbAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        train, val, test = values
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            parser.error(
                f"--split-probabilities must sum to 1.0; "
                f"got {train:.3f} + {val:.3f} + {test:.3f} = {total:.3f}"
            )
        setattr(namespace, self.dest, values)
