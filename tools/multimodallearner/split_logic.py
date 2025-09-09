import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_COLUMN_NAME = 'split'  # 0=train, 1=val, 2=test


# ---------- Small utilities ----------

def path_expander(p: str, base_folder: str) -> str:
    """Make relative paths absolute against base_folder."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_folder, p))


def map_column_arg(col_arg: Optional[str], df: pd.DataFrame, arg_name: str) -> Optional[str]:
    """Support Galaxy-style numeric column arguments ('6' -> 5th index)."""
    if col_arg and str(col_arg).isdigit():
        col_idx = int(col_arg) - 1  # Galaxy columns start at 1
        if col_idx < 0 or col_idx >= len(df.columns):
            sys.stderr.write(
                f'ERROR: Invalid {arg_name} index {col_arg} (out of range 1-{len(df.columns)})\n'
            )
            sys.exit(1)
        mapped = df.columns[col_idx]
        return mapped
    return col_arg


# ---------- Split helpers (Image Learner–style) ----------

def split_data_0_2(
    df: pd.DataFrame,
    split_column: str,
    validation_size: float = 0.125,
    random_state: int = 42,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame whose split_column only contains {0,2},
    re-assign a portion of the 0s to become 1s (validation).
    Uses stratification if possible.
    """
    out = df.copy()
    out[split_column] = pd.to_numeric(out[split_column], errors='coerce').astype(int)
    idx_train = out.index[out[split_column] == 0].tolist()
    if not idx_train:
        return out

    stratify_arr = None
    if label_column and label_column in out.columns:
        vc = out.loc[idx_train, label_column].value_counts()
        if vc.size > 1:
            # ensure at least one sample/class in validation
            min_per_class = vc.min()
            if min_per_class * validation_size < 1:
                validation_size = min(validation_size, 1.0 / max(1, min_per_class))
            stratify_arr = out.loc[idx_train, label_column]

    try:
        tr_idx, va_idx = train_test_split(
            idx_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify_arr,
        )
    except ValueError:
        tr_idx, va_idx = train_test_split(
            idx_train, test_size=validation_size, random_state=random_state, stratify=None
        )

    out.loc[tr_idx, split_column] = 0
    out.loc[va_idx, split_column] = 1
    out[split_column] = out[split_column].astype(int)
    return out


def _random_assign(
    out: pd.DataFrame,
    split_column: str,
    probs: List[float],
    random_state: int,
) -> pd.DataFrame:
    idx = out.index.to_list()
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(n * probs[0])
    n_va = int(n * probs[1])
    out.loc[idx[:n_tr], split_column] = 0
    out.loc[idx[n_tr:n_tr + n_va], split_column] = 1
    out.loc[idx[n_tr + n_va:], split_column] = 2
    return out.astype({split_column: int})


def create_stratified_random_split(
    df: pd.DataFrame,
    split_column: str,
    split_probabilities: List[float],
    random_state: int,
    label_column: Optional[str],
) -> pd.DataFrame:
    """
    Create a stratified random split (train/val/test) if feasible.
    Falls back to random assignment if class counts are too small.
    """
    out = df.copy()
    out[split_column] = 0

    if not label_column or label_column not in out.columns:
        return _random_assign(out, split_column, split_probabilities, random_state)

    vc = out[label_column].value_counts()
    if vc.empty or vc.min() < 3:  # need at least 3 samples per class for 3 splits
        return _random_assign(out, split_column, split_probabilities, random_state)

    # 1) hold out test
    tr_va_idx, te_idx = train_test_split(
        out.index.to_list(),
        test_size=split_probabilities[2],
        random_state=random_state,
        stratify=out[label_column],
    )
    # 2) split train/val from train+val
    val_size_adj = split_probabilities[1] / (split_probabilities[0] + split_probabilities[1])
    tr_idx, va_idx = train_test_split(
        tr_va_idx,
        test_size=val_size_adj,
        random_state=random_state,
        stratify=out.loc[tr_va_idx, label_column],
    )

    out.loc[tr_idx, split_column] = 0
    out.loc[va_idx, split_column] = 1
    out.loc[te_idx, split_column] = 2
    return out.astype({split_column: int})


# ---------- Unified loader + splitter ----------

def load_and_split(
    train_csv: str,
    test_csv: Optional[str],
    label_column: str,
    image_column: Optional[str],
    random_seed: int,
    validation_size: float,
    split_probabilities: List[float],
    val_size_with_test: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Optional[str]]:
    """
    Load CSV(s) and return (df_train, df_val, df_test, df_train_full, label_col, image_col),
    mapping numeric column args if needed and applying split rules:
      - If test_csv provided → stratified train/val split with val_size_with_test
      - Else if split column exists:
          * If {0,2} → reassign part of 0 to 1 using validation_size (stratified if possible)
          * If {0,1,2} → use as-is
      - Else → create stratified random split using split_probabilities
    """
    if test_csv:
        df_train_full = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        label_column = map_column_arg(label_column, df_train_full, 'target_column')
        image_column = map_column_arg(image_column, df_train_full, 'image_column')

        vc = df_train_full[label_column].value_counts()
        stratify_col = df_train_full[label_column] if (vc.min() >= 2) else None

        df_train, df_val = train_test_split(
            df_train_full,
            test_size=val_size_with_test,
            random_state=random_seed,
            stratify=stratify_col,
        )
        return df_train, df_val, df_test, df_train_full, label_column, image_column

    # No test CSV → single CSV with either fixed split or random split
    df_full = pd.read_csv(train_csv)
    label_column = map_column_arg(label_column, df_full, 'target_column')
    image_column = map_column_arg(image_column, df_full, 'image_column')

    if SPLIT_COLUMN_NAME in df_full.columns:
        df_full[SPLIT_COLUMN_NAME] = pd.to_numeric(df_full[SPLIT_COLUMN_NAME], errors='coerce').astype('Int64')
        unique_vals = set(df_full[SPLIT_COLUMN_NAME].dropna().unique())

        if unique_vals == {0, 2}:
            df_full = split_data_0_2(
                df=df_full,
                split_column=SPLIT_COLUMN_NAME,
                validation_size=validation_size,
                random_state=random_seed,
                label_column=label_column,
            )
        elif not unique_vals.issubset({0, 1, 2}):
            sys.stderr.write(f'ERROR: Unexpected split values: {unique_vals}\n')
            sys.exit(1)

        df_train = df_full[df_full[SPLIT_COLUMN_NAME] == 0].copy()
        df_val   = df_full[df_full[SPLIT_COLUMN_NAME] == 1].copy()
        df_test  = df_full[df_full[SPLIT_COLUMN_NAME] == 2].copy()
        df_train_full = df_full[df_full[SPLIT_COLUMN_NAME].isin([0, 1])].copy()

        if len(df_val) == 0 or len(df_test) == 0:
            sys.stderr.write('ERROR: Empty validation or test set after fixed split processing.\n')
            sys.exit(1)
        return df_train, df_val, df_test, df_train_full, label_column, image_column

    # No split column → random stratified
    df_split = create_stratified_random_split(
        df=df_full,
        split_column=SPLIT_COLUMN_NAME,
        split_probabilities=split_probabilities,
        random_state=random_seed,
        label_column=label_column,
    )
    df_train = df_split[df_split[SPLIT_COLUMN_NAME] == 0].copy()
    df_val   = df_split[df_split[SPLIT_COLUMN_NAME] == 1].copy()
    df_test  = df_split[df_split[SPLIT_COLUMN_NAME] == 2].copy()
    df_train_full = df_split[df_split[SPLIT_COLUMN_NAME].isin([0, 1])].copy()
    return df_train, df_val, df_test, df_train_full, label_column, image_column
