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
) -> pd.DataFrame:
    """Create a stratified random split when no split column exists."""
    out = df.copy()

    # initialize split column
    out[split_column] = 0

    if not label_column or label_column not in out.columns:
        logger.warning(
            "No label column found; using random split without stratification"
        )
        # fall back to simple random assignment
        indices = out.index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * split_probabilities[0])
        n_val = int(n_total * split_probabilities[1])

        out.loc[indices[:n_train], split_column] = 0
        out.loc[indices[n_train:n_train + n_val], split_column] = 1
        out.loc[indices[n_train + n_val:], split_column] = 2

        return out.astype({split_column: int})

    # check if stratification is possible
    label_counts = out[label_column].value_counts()
    min_samples_per_class = label_counts.min()

    # ensure we have enough samples for stratification:
    # Each class must have at least as many samples as the number of splits,
    # so that each split can receive at least one sample per class.
    min_samples_required = len(split_probabilities)
    if min_samples_per_class < min_samples_required:
        logger.warning(
            f"Insufficient samples per class for stratification (min: {min_samples_per_class}, required: {min_samples_required}); using random split"
        )
        # fall back to simple random assignment
        indices = out.index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * split_probabilities[0])
        n_val = int(n_total * split_probabilities[1])

        out.loc[indices[:n_train], split_column] = 0
        out.loc[indices[n_train:n_train + n_val], split_column] = 1
        out.loc[indices[n_train + n_val:], split_column] = 2

        return out.astype({split_column: int})

    logger.info("Using stratified random split for train/validation/test sets")

    # first split: separate test set
    train_val_idx, test_idx = train_test_split(
        out.index.tolist(),
        test_size=split_probabilities[2],
        random_state=random_state,
        stratify=out[label_column],
    )

    # second split: separate training and validation from remaining data
    val_size_adjusted = split_probabilities[1] / (
        split_probabilities[0] + split_probabilities[1]
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=out.loc[train_val_idx, label_column] if label_column and label_column in out.columns else None,
    )

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
