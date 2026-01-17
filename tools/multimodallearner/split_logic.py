import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
SPLIT_COL = "split"


def _can_stratify(y: pd.Series) -> bool:
    return y.nunique() >= 2 and (y.value_counts() >= 2).all()


def split_dataset(
    train_dataset: pd.DataFrame,
    test_dataset: Optional[pd.DataFrame],
    target_column: str,
    split_probabilities: List[float],
    validation_size: float,
    random_seed: int = 42,
    sample_id_column: Optional[str] = None,
) -> None:
    if target_column not in train_dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Drop NaN labels early
    before = len(train_dataset)
    train_dataset.dropna(subset=[target_column], inplace=True)
    if len(train_dataset) == 0:
        raise ValueError("No rows remain after dropping NaN targets")
    if before != len(train_dataset):
        logger.warning(f"Dropped {before - len(train_dataset)} rows with NaN target")
    y = train_dataset[target_column]

    if sample_id_column and sample_id_column not in train_dataset.columns:
        logger.warning(
            "Sample ID column '%s' not found; proceeding without group-aware split.",
            sample_id_column,
        )
        sample_id_column = None
    if sample_id_column and sample_id_column == target_column:
        logger.warning(
            "Sample ID column '%s' matches target column; proceeding without group-aware split.",
            sample_id_column,
        )
        sample_id_column = None

    # Respect existing valid split column
    if SPLIT_COL in train_dataset.columns:
        unique = set(train_dataset[SPLIT_COL].dropna().unique())
        valid = {"train", "val", "validation", "test"}
        if unique.issubset(valid):
            train_dataset[SPLIT_COL] = train_dataset[SPLIT_COL].replace("validation", "val")
            normalized = set(train_dataset[SPLIT_COL].dropna().unique())
            required = {"train"} if test_dataset is not None else {"train", "test"}
            missing = required - normalized
            if missing:
                missing_list = ", ".join(sorted(missing))
                if test_dataset is not None:
                    raise ValueError(
                        "Pre-existing 'split' column is missing required split(s): "
                        f"{missing_list}. Expected at least train when an external test set is provided, "
                        "or remove the 'split' column to let the tool create splits."
                    )
                raise ValueError(
                    "Pre-existing 'split' column is missing required split(s): "
                    f"{missing_list}. Expected at least train and test, "
                    "or remove the 'split' column to let the tool create splits."
                )
            logger.info(f"Using pre-existing 'split' column: {sorted(normalized)}")
            return

    train_dataset[SPLIT_COL] = "train"

    def _allocate_split_counts(n_total: int, probs: list) -> list:
        if n_total <= 0:
            return [0 for _ in probs]
        counts = [0 for _ in probs]
        active = [i for i, p in enumerate(probs) if p > 0]
        remainder = n_total
        if active and n_total >= len(active):
            for i in active:
                counts[i] = 1
            remainder -= len(active)
        if remainder > 0:
            probs_arr = np.array(probs, dtype=float)
            probs_arr = probs_arr / probs_arr.sum()
            raw = remainder * probs_arr
            floors = np.floor(raw).astype(int)
            for i, value in enumerate(floors.tolist()):
                counts[i] += value
            leftover = remainder - int(floors.sum())
            if leftover > 0 and active:
                frac = raw - floors
                order = sorted(active, key=lambda i: (-frac[i], i))
                for i in range(leftover):
                    counts[order[i % len(order)]] += 1
        return counts

    def _choose_split(counts: list, targets: list, active: list) -> int:
        remaining = [targets[i] - counts[i] for i in range(len(targets))]
        best = max(active, key=lambda i: (remaining[i], -counts[i], -targets[i]))
        if remaining[best] <= 0:
            best = min(active, key=lambda i: counts[i])
        return best

    if test_dataset is not None:
        if sample_id_column and sample_id_column in test_dataset.columns:
            train_ids = set(train_dataset[sample_id_column].dropna().astype(object).unique())
            test_ids = set(test_dataset[sample_id_column].dropna().astype(object).unique())
            overlap = train_ids & test_ids
            if overlap:
                logger.warning(
                    "Sample ID column '%s' has %d overlapping IDs between train and external test sets; "
                    "consider removing overlaps to avoid leakage.",
                    sample_id_column,
                    len(overlap),
                )
        if sample_id_column:
            rng = np.random.RandomState(random_seed)
            group_series = train_dataset[sample_id_column].astype(object)
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
            targets = _allocate_split_counts(len(train_dataset), [1.0 - validation_size, validation_size])
            counts = [0, 0]
            active = [0, 1]
            train_idx = []
            val_idx = []
            for group_id in group_ids:
                split_idx = _choose_split(counts, targets, active)
                counts[split_idx] += len(group_to_indices[group_id])
                if split_idx == 0:
                    train_idx.extend(group_to_indices[group_id])
                else:
                    val_idx.extend(group_to_indices[group_id])
            train_dataset.loc[val_idx, SPLIT_COL] = "val"
        else:
            if validation_size <= 0:
                logger.warning(
                    "validation_size is %.3f; skipping validation split to avoid train_test_split errors.",
                    validation_size,
                )
            elif validation_size >= 1:
                logger.warning(
                    "validation_size is %.3f; assigning all rows to validation to avoid train_test_split errors.",
                    validation_size,
                )
                train_dataset[SPLIT_COL] = "val"
            else:
                stratify = y if _can_stratify(y) else None
                train_idx, val_idx = train_test_split(
                    train_dataset.index, test_size=validation_size,
                    random_state=random_seed, stratify=stratify
                )
                train_dataset.loc[val_idx, SPLIT_COL] = "val"
        logger.info(f"External test set → created val split ({validation_size:.0%})")

    else:
        p_train, p_val, p_test = split_probabilities
        if abs(p_train + p_val + p_test - 1.0) > 1e-6:
            raise ValueError("split_probabilities must sum to 1.0")

        if sample_id_column:
            rng = np.random.RandomState(random_seed)
            group_series = train_dataset[sample_id_column].astype(object)
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
            targets = _allocate_split_counts(len(train_dataset), [p_train, p_val, p_test])
            counts = [0, 0, 0]
            active = [0, 1, 2]
            train_idx = []
            val_idx = []
            test_idx = []
            for group_id in group_ids:
                split_idx = _choose_split(counts, targets, active)
                counts[split_idx] += len(group_to_indices[group_id])
                if split_idx == 0:
                    train_idx.extend(group_to_indices[group_id])
                elif split_idx == 1:
                    val_idx.extend(group_to_indices[group_id])
                else:
                    test_idx.extend(group_to_indices[group_id])
            train_dataset.loc[val_idx, SPLIT_COL] = "val"
            train_dataset.loc[test_idx, SPLIT_COL] = "test"
        else:
            stratify = y if _can_stratify(y) else None
            if p_test <= 0:
                logger.warning(
                    "split_probabilities specify 0 test size; skipping test split to avoid train_test_split errors."
                )
                tv_idx = train_dataset.index
                test_idx = train_dataset.index[:0]
            elif p_test >= 1:
                logger.warning(
                    "split_probabilities specify 100% test size; assigning all rows to test to avoid train_test_split errors."
                )
                tv_idx = train_dataset.index[:0]
                test_idx = train_dataset.index
            else:
                tv_idx, test_idx = train_test_split(
                    train_dataset.index, test_size=p_test,
                    random_state=random_seed, stratify=stratify
                )
            rel_val = p_val / (p_train + p_val) if (p_train + p_val) > 0 else 0
            train_idx = train_dataset.index[:0]
            val_idx = train_dataset.index[:0]
            if len(tv_idx):
                if rel_val <= 0:
                    logger.warning(
                        "split_probabilities specify 0 validation size; skipping validation split to avoid train_test_split errors."
                    )
                    train_idx = tv_idx
                elif rel_val >= 1:
                    logger.warning(
                        "split_probabilities specify 100% validation size; assigning all remaining rows to validation "
                        "to avoid train_test_split errors."
                    )
                    val_idx = tv_idx
                else:
                    strat_tv = y.loc[tv_idx] if _can_stratify(y.loc[tv_idx]) else None
                    train_idx, val_idx = train_test_split(
                        tv_idx, test_size=rel_val,
                        random_state=random_seed, stratify=strat_tv
                    )

            train_dataset.loc[val_idx, SPLIT_COL] = "val"
            train_dataset.loc[test_idx, SPLIT_COL] = "test"
        logger.info(f"3-way split → train:{len(train_idx)}, val:{len(val_idx)}, test:{len(test_idx)}")

    logger.info(f"Final split distribution:\n{train_dataset[SPLIT_COL].value_counts().sort_index()}")
