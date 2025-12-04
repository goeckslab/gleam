import logging
from typing import List, Optional

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
) -> None:
    if target_column not in train_dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    y = train_dataset[target_column]

    # Drop NaN labels early
    before = len(train_dataset)
    train_dataset.dropna(subset=[target_column], inplace=True)
    if len(train_dataset) == 0:
        raise ValueError("No rows remain after dropping NaN targets")
    if before != len(train_dataset):
        logger.warning(f"Dropped {before - len(train_dataset)} rows with NaN target")

    # Respect existing valid split column
    if SPLIT_COL in train_dataset.columns:
        unique = set(train_dataset[SPLIT_COL].dropna().unique())
        valid = {"train", "val", "validation", "test"}
        if unique.issubset(valid | {"validation"}):
            train_dataset[SPLIT_COL] = train_dataset[SPLIT_COL].replace("validation", "val")
            logger.info(f"Using pre-existing 'split' column: {sorted(unique)}")
            return

    train_dataset[SPLIT_COL] = "train"

    if test_dataset is not None:
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

        stratify = y if _can_stratify(y) else None
        tv_idx, test_idx = train_test_split(
            train_dataset.index, test_size=p_test,
            random_state=random_seed, stratify=stratify
        )
        rel_val = p_val / (p_train + p_val) if (p_train + p_val) > 0 else 0
        strat_tv = y.loc[tv_idx] if _can_stratify(y.loc[tv_idx]) else None
        train_idx, val_idx = train_test_split(
            tv_idx, test_size=rel_val,
            random_state=random_seed, stratify=strat_tv
        )

        train_dataset.loc[val_idx, SPLIT_COL] = "val"
        train_dataset.loc[test_idx, SPLIT_COL] = "test"
        logger.info(f"3-way split → train:{len(train_idx)}, val:{len(val_idx)}, test:{len(test_idx)}")

    logger.info(f"Final split distribution:\n{train_dataset[SPLIT_COL].value_counts().sort_index()}")
