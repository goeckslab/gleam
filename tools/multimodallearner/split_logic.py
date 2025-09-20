from __future__ import annotations

import os
import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Public constants expected by other modules
SPLIT_COLUMN_NAME = "split"

logger = logging.getLogger(__name__)


# ---------- Path expanders ----------
def path_expander(p: str, base_folder: str) -> str:
    """Legacy single-base expander. Kept for back-compat."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_folder, p))


def path_expander_any(p: str, base_folders: Sequence[str]) -> str:
    """
    Resolve p against many candidate base folders.

    Strategy:
      1) If absolute, return as-is.
      2) Try base/<p>.
      3) If not found, recursively search base/**/<basename(p)> and return first hit.
      4) Fallback to base_folders[0]/p, or CWD/p if no bases provided.
    """
    if os.path.isabs(p):
        return p
    basename = os.path.basename(str(p))
    for base in (base_folders or []):
        cand = os.path.abspath(os.path.join(base, p))
        if os.path.exists(cand):
            return cand
        # recursive search for basename (helps when zips extract into nested dirs)
        for root, _dirs, files in os.walk(base):
            if basename in files:
                return os.path.abspath(os.path.join(root, basename))
    if base_folders:
        return os.path.abspath(os.path.join(base_folders[0], p))
    return os.path.abspath(p)


# ---------- Column mapping helpers ----------
def _as_int_like(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def map_column_arg(arg: Union[str, int], df: pd.DataFrame, arg_name: str) -> str:
    """
    Map a single column arg (name or 1-based index) to a real column name.
    - If int-like → treat as 1-based index (common in Galaxy tools).
    - Else → treat as column name.
    """
    if isinstance(arg, int):
        idx = arg
    else:
        si = _as_int_like(str(arg))
        if si is not None:
            idx = si
        else:
            name = str(arg)
            if name in df.columns:
                return name
            raise ValueError(f"{arg_name}: column '{name}' not found in DataFrame.")

    # int branch → 1-based index
    if idx < 1 or idx > len(df.columns):
        raise ValueError(f"{arg_name}: column index {idx} out of range 1..{len(df.columns)}.")
    return str(df.columns[idx - 1])


def map_column_args(
    col_args: Optional[Union[str, Sequence[Union[str, int]]]],
    df: pd.DataFrame,
    arg_name: str,
) -> Optional[List[str]]:
    """
    Map one or many column args (names or 1-based indices) to column names.
    Accepts 'col1,col2' OR ['col1','col2'] OR [3,5] etc.
    """
    if col_args is None:
        return None
    if isinstance(col_args, str):
        parts: List[Union[str, int]] = [s.strip() for s in col_args.split(",") if s.strip()]
    else:
        parts = list(col_args)
    mapped: List[str] = []
    for c in parts:
        mapped.append(map_column_arg(c, df, arg_name))
    return mapped


# ---------- Label cleaning & stratification helpers ----------
def _log_class_counts(df: pd.DataFrame, label_col: str, context: str) -> None:
    vc = df[label_col].value_counts(dropna=False)
    logger.info(f"[{context}] label '{label_col}' value counts (including NaN):\n{vc.to_string()}")


def _drop_rows_with_nan_label(df: pd.DataFrame, label_col: str, context: str) -> pd.DataFrame:
    """
    Drop rows where the target is NaN and log how many were dropped.
    """
    total = len(df)
    nan_count = int(df[label_col].isna().sum())
    logger.info(f"[{context}] rows: {total} | NaNs in label '{label_col}': {nan_count}")
    if nan_count == 0:
        logger.info(f"[{context}] dropped 0 rows (no NaN targets).")
        return df.copy()

    df2 = df[df[label_col].notna()].copy()
    dropped = total - len(df2)
    logger.warning(f"[{context}] dropped {dropped} rows with NaN in '{label_col}'.")
    if len(df2) == 0:
        raise ValueError(f"After dropping NaN targets in {context}, no rows remain.")
    return df2


def _can_stratify(df: pd.DataFrame, label_col: str) -> bool:
    """Return True iff there are at least two classes with count >= 2 (ignoring NaN)."""
    if label_col not in df.columns:
        return False
    s_non_na = df[label_col].dropna()
    if s_non_na.empty:
        return False
    vc = s_non_na.value_counts()
    return len(vc) >= 2 and vc.min() >= 2


# ---------- Splitting helpers ----------
def _random_stratified_split(
    df_full: pd.DataFrame,
    label_col: str,
    split_probabilities: List[float],
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    No split column → create random stratified Train/Val/Test according to probabilities.
    Assumes df_full already has NaN targets removed.
    """
    p_train, p_val, p_test = split_probabilities
    if abs(p_train + p_val + p_test - 1.0) > 1e-6:
        raise ValueError("split_probabilities must sum to 1.0")

    _log_class_counts(df_full, label_col, "pre-random-split (full)")
    # First carve out Test
    strat = df_full[label_col] if _can_stratify(df_full, label_col) else None
    df_hold, df_test = train_test_split(
        df_full, test_size=p_test, random_state=int(random_seed), stratify=strat
    )
    _log_class_counts(df_test, label_col, "post-test-split (test)")

    # Then split hold into Train/Val by relative proportions
    remain = p_train + p_val
    if remain <= 0:
        # Degenerate case: no train/val requested
        df_train = df_full.iloc[0:0].copy()
        df_val = df_full.iloc[0:0].copy()
        df_train_full = df_full.iloc[0:0].copy()
    else:
        rel_val = p_val / remain
        strat2 = df_hold[label_col] if _can_stratify(df_hold, label_col) else None
        df_train, df_val = train_test_split(
            df_hold, test_size=rel_val, random_state=int(random_seed), stratify=strat2
        )
        df_train_full = pd.concat([df_train, df_val], axis=0)
        _log_class_counts(df_train, label_col, "post-holdout-split (train)")
        _log_class_counts(df_val, label_col,   "post-holdout-split (val)")

    return df_train, df_val, df_test, df_train_full


def _split_from_fixed_column(
    df_full: pd.DataFrame,
    label_col: str,
    validation_size: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Use existing SPLIT_COLUMN_NAME with values in {0,1,2}.
    If only {0,2} are present, create Validation from Train(0) by holding out 'validation_size'.
    Assumes df_full already has NaN targets removed.
    """
    s = pd.to_numeric(df_full[SPLIT_COLUMN_NAME], errors="coerce").astype("Int64")
    df_full = df_full.copy()
    df_full[SPLIT_COLUMN_NAME] = s

    uniq = set(s.dropna().unique().tolist())
    has0 = 0 in uniq
    has1 = 1 in uniq
    has2 = 2 in uniq

    if has0 and has2 and not has1:
        # Create validation from the train(0) partition
        df_tr = df_full[df_full[SPLIT_COLUMN_NAME] == 0].copy()
        df_te = df_full[df_full[SPLIT_COLUMN_NAME] == 2].copy()

        _log_class_counts(df_tr, label_col, "fixed-split {0,2} (train0)")
        _log_class_counts(df_te, label_col, "fixed-split {0,2} (test2)")

        val_size = float(validation_size)
        if not (0.0 < val_size < 1.0):
            raise ValueError("--validation_size must be in (0,1) when creating validation from {0,2} split.")

        strat = df_tr[label_col] if _can_stratify(df_tr, label_col) else None
        df_tr2, df_va = train_test_split(
            df_tr, test_size=val_size, random_state=int(random_seed), stratify=strat
        )
        df_train = df_tr2
        df_val = df_va
        df_test = df_te
        df_train_full = pd.concat([df_train, df_val], axis=0)

        _log_class_counts(df_train, label_col, "after-val-from-train0 (train)")
        _log_class_counts(df_val,   label_col, "after-val-from-train0 (val)")

    else:
        # Use provided assignments
        df_train = df_full[df_full[SPLIT_COLUMN_NAME] == 0].copy()
        df_val = df_full[df_full[SPLIT_COLUMN_NAME] == 1].copy()
        df_test = df_full[df_full[SPLIT_COLUMN_NAME] == 2].copy()
        df_train_full = df_full[df_full[SPLIT_COLUMN_NAME].isin([0, 1])].copy()

        _log_class_counts(df_train, label_col, "fixed-split (train)")
        _log_class_counts(df_val,   label_col, "fixed-split (val)")
        _log_class_counts(df_test,  label_col, "fixed-split (test)")

        # Guardrails: if val is empty, carve a small one from train
        if len(df_val) == 0 and len(df_train) > 1:
            strat = df_train[label_col] if _can_stratify(df_train, label_col) else None
            df_train, df_val = train_test_split(
                df_train, test_size=0.125, random_state=int(random_seed), stratify=strat
            )
            df_train_full = pd.concat([df_train, df_val], axis=0)
            _log_class_counts(df_train, label_col, "val-created-from-train (train)")
            _log_class_counts(df_val,   label_col, "val-created-from-train (val)")

    return df_train, df_val, df_test, df_train_full


# ---------- Main entry ----------
def load_and_split(
    train_csv: str,
    test_csv: Optional[str],
    label_column: str,
    image_columns: Optional[List[str]],  # list, not single
    random_seed: int,
    validation_size: float,
    split_probabilities: List[float],
    val_size_with_test: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Optional[List[str]]]:
    """
    Reads CSV(s), maps column arguments, drops rows with NaN targets (with logs), and returns:
      (df_train, df_val, df_test, df_train_full, label_col, image_cols)

    Behavior:
      - If 'split' column exists: use it; if only {0,2}, create validation from 0's using validation_size.
      - Else: create random stratified split via split_probabilities.
      - If a separate test_csv is provided: split train_csv into (train/val) by val_size_with_test and use test_csv as test.
    """
    if test_csv:
        # Two-file mode: split train_csv into train/val; test=external CSV
        df_train_full = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        label_col = map_column_arg(label_column, df_train_full, "target_column")
        img_cols = map_column_args(image_columns, df_train_full, "image_columns")

        logger.info(f"[load] resolved label column: '{label_col}' | dtype={df_train_full[label_col].dtype}")
        # Pre-drop logs
        _log_class_counts(df_train_full, label_col, "train_csv (pre-drop)")
        _log_class_counts(df_test,       label_col, "test_csv (pre-drop)")

        # Drop rows with NaN label (both train side and test side) with logs
        df_train_full = _drop_rows_with_nan_label(df_train_full, label_col, "train_csv")
        df_test = _drop_rows_with_nan_label(df_test, label_col, "test_csv")

        _log_class_counts(df_train_full, label_col, "train_csv (post-drop)")
        _log_class_counts(df_test,       label_col, "test_csv (post-drop)")

        # Stratify if possible (on cleaned labels)
        strat = df_train_full[label_col] if _can_stratify(df_train_full, label_col) else None
        df_train, df_val = train_test_split(
            df_train_full,
            test_size=float(val_size_with_test),
            random_state=int(random_seed),
            stratify=strat,
        )
        _log_class_counts(df_train, label_col, "train after split")
        _log_class_counts(df_val,   label_col, "val   after split")

        return df_train, df_val, df_test, df_train_full, label_col, img_cols

    # Single-file mode
    df_full = pd.read_csv(train_csv)

    label_col = map_column_arg(label_column, df_full, "target_column")
    img_cols = map_column_args(image_columns, df_full, "image_columns")

    logger.info(f"[load] resolved label column: '{label_col}' | dtype={df_full[label_col].dtype}")
    _log_class_counts(df_full, label_col, "train_csv (pre-drop)")

    # Drop rows with NaN label before any splitting (with logs)
    df_full = _drop_rows_with_nan_label(df_full, label_col, "train_csv")
    _log_class_counts(df_full, label_col, "train_csv (post-drop)")

    if SPLIT_COLUMN_NAME in df_full.columns:
        df_train, df_val, df_test, df_train_full = _split_from_fixed_column(
            df_full, label_col, float(validation_size), int(random_seed)
        )

        # If val/test ended empty (pathological inputs), fall back to random split
        if len(df_val) == 0 or len(df_test) == 0:
            logger.warning("Empty validation or test detected; falling back to random stratified split.")
            df_train, df_val, df_test, df_train_full = _random_stratified_split(
                df_full, label_col, split_probabilities, int(random_seed)
            )
    else:
        df_train, df_val, df_test, df_train_full = _random_stratified_split(
            df_full, label_col, split_probabilities, int(random_seed)
        )

    return df_train, df_val, df_test, df_train_full, label_col, img_cols