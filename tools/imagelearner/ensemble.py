"""Cross-validation and multi-seed ensembles for Image Learner.

An ensemble run trains several Ludwig *members* that differ only in their
split column (k-fold cross-validation) and/or random seed. Members are trained
one after another and their per-row predictions are averaged by base-row
index, so every row is scored only by models that never trained on it.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from constants import (
    DECISION_THRESHOLD_COLUMN,
    LABEL_COLUMN_NAME,
    SPLIT_COLUMN_NAME,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from split_data import create_stratified_random_split
from utils import detect_output_type, extract_metrics_from_json, extract_test_metrics, nonzero_binary_label

logger = logging.getLogger("ImageLearner")

FOLD_ROLE_VALIDATION = "validation"
FOLD_ROLE_TEST = "test"
FOLD_ROLES = (FOLD_ROLE_VALIDATION, FOLD_ROLE_TEST)

PREDICTIONS_COLUMN = "label_predictions"
PROBABILITY_COLUMN = "label_probability"
PROBABILITIES_VECTOR_COLUMN = "label_probabilities"
PROBABILITIES_PREFIX = "label_probabilities_"
N_MODELS_COLUMN = "n_models"

ENSEMBLE_SUMMARY_FILE_NAME = "ensemble_summary.json"
MEMBERS_DIR_NAME = "members"

_POSITIVE_CLASS_HINTS = ("event", "true", "positive", "pos", "1")


@dataclass
class MemberSpec:
    """One model of the ensemble: a seed plus the split column it trains on."""

    name: str
    seed: int
    fold: Optional[int]
    split: pd.Series


def ensemble_enabled(cross_validation: bool, num_seeds: int) -> bool:
    return bool(cross_validation) or int(num_seeds or 1) > 1


def assign_folds(
    df: pd.DataFrame,
    num_folds: int,
    random_state: int,
    label_column: Optional[str] = None,
    group_column: Optional[str] = None,
) -> np.ndarray:
    """Return a 0-based fold id for every row of ``df``.

    Rows are stratified by ``label_column`` and kept together by
    ``group_column`` when those columns are given. If stratification is
    impossible (for example every class has fewer rows than folds) the
    assignment falls back to plain (group) k-fold with a warning.
    """
    num_folds = int(num_folds)
    n_rows = len(df)
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 for cross-validation.")
    if n_rows < num_folds:
        raise ValueError(f"Cannot create {num_folds} folds from {n_rows} rows.")

    groups = None
    if group_column and group_column in df.columns:
        group_series = df[group_column].astype(object).copy()
        missing_mask = group_series.isna()
        if missing_mask.any():
            group_series.loc[missing_mask] = [
                f"__missing__{idx}" for idx in group_series.index[missing_mask]
            ]
        groups = group_series.astype(str).to_numpy()
        n_groups = len(set(groups))
        if n_groups < num_folds:
            raise ValueError(
                f"Cannot create {num_folds} folds from {n_groups} unique values "
                f"of the sample ID column '{group_column}'."
            )

    y = None
    if label_column and label_column in df.columns:
        y = df[label_column].astype(str).to_numpy()

    placeholder = np.zeros((n_rows, 1))
    folds = np.full(n_rows, -1, dtype=int)

    def _fill(splitter, y_arg, groups_arg) -> None:
        folds[:] = -1
        for fold_id, (_, held_out) in enumerate(splitter.split(placeholder, y_arg, groups_arg)):
            folds[held_out] = fold_id

    if y is not None:
        try:
            if groups is not None:
                splitter = StratifiedGroupKFold(
                    n_splits=num_folds, shuffle=True, random_state=random_state
                )
            else:
                splitter = StratifiedKFold(
                    n_splits=num_folds, shuffle=True, random_state=random_state
                )
            _fill(splitter, y, groups)
            if (folds >= 0).all():
                return folds
        except ValueError as exc:
            logger.warning(
                "Stratified %d-fold assignment failed (%s); using unstratified folds.",
                num_folds,
                exc,
            )

    if groups is not None:
        splitter = GroupKFold(n_splits=num_folds)
    else:
        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    _fill(splitter, None, groups)
    return folds


def build_member_plan(
    df: pd.DataFrame,
    *,
    cross_validation: bool,
    num_folds: int,
    fold_role: str,
    num_seeds: int,
    random_seed: int,
    validation_size: float,
    label_column: str = LABEL_COLUMN_NAME,
    group_column: Optional[str] = None,
    stratify: bool = True,
    split_column: str = SPLIT_COLUMN_NAME,
) -> List[MemberSpec]:
    """Plan the members of an ensemble from a prepared dataframe.

    ``df`` must already carry a 0/1/2 split column. Without cross-validation
    every member reuses that split and only the seed changes. With
    cross-validation the folds are drawn once (seeded by ``random_seed``) and
    every seed is trained on every fold, so members = folds x seeds.

    fold_role "validation" keeps the test rows fixed and rotates the folds of
    the remaining rows through the validation set. fold_role "test" rotates
    the folds of *all* rows through the test set and draws a validation set
    of ``validation_size`` from each fold's training rows.
    """
    if split_column not in df.columns:
        raise ValueError(f"Prepared data has no '{split_column}' column.")
    base_split = pd.to_numeric(df[split_column], errors="coerce").fillna(0).astype(int)
    seeds = [int(random_seed) + offset for offset in range(max(1, int(num_seeds)))]
    strat_column = label_column if stratify and label_column in df.columns else None

    if not cross_validation:
        return [
            MemberSpec(name=f"seed{seed}", seed=seed, fold=None, split=base_split.copy())
            for seed in seeds
        ]

    if fold_role not in FOLD_ROLES:
        raise ValueError(f"fold_role must be one of {FOLD_ROLES}; got '{fold_role}'.")
    num_folds = int(num_folds)
    members: List[MemberSpec] = []

    if fold_role == FOLD_ROLE_VALIDATION:
        pool_mask = base_split != 2
        pool = df.loc[pool_mask]
        if pool.empty:
            raise ValueError(
                "No training rows available for cross-validation: every row is in the test split."
            )
        fold_ids = pd.Series(
            assign_folds(pool, num_folds, random_seed, strat_column, group_column),
            index=pool.index,
        )
        for fold in range(num_folds):
            split = base_split.copy()
            split.loc[pool_mask] = 0
            split.loc[fold_ids.index[fold_ids == fold]] = 1
            for seed in seeds:
                members.append(
                    MemberSpec(
                        name=f"fold{fold + 1}_seed{seed}",
                        seed=seed,
                        fold=fold + 1,
                        split=split.copy(),
                    )
                )
        return members

    if (base_split == 2).any():
        logger.info(
            "Fold role 'test': existing test rows are folded back into the cross-validation pool."
        )
    fold_ids = pd.Series(
        assign_folds(df, num_folds, random_seed, strat_column, group_column),
        index=df.index,
    )
    for fold in range(num_folds):
        split = pd.Series(0, index=df.index, dtype=int)
        split.loc[fold_ids.index[fold_ids == fold]] = 2
        train_rows = df.loc[split == 0]
        if validation_size > 0 and len(train_rows) > 1:
            carved = create_stratified_random_split(
                train_rows,
                split_column,
                [1.0 - float(validation_size), float(validation_size), 0.0],
                random_state=int(random_seed) + fold,
                label_column=strat_column,
                group_column=group_column,
            )
            split.loc[carved.index] = carved[split_column].astype(int)
        else:
            logger.warning(
                "Fold %d has no validation rows (validation_size=%s); Ludwig will train without early stopping data.",
                fold + 1,
                validation_size,
            )
        for seed in seeds:
            members.append(
                MemberSpec(
                    name=f"fold{fold + 1}_seed{seed}",
                    seed=seed,
                    fold=fold + 1,
                    split=split.copy(),
                )
            )
    return members


def canonical_label(value: Any) -> str:
    """A label as the string Ludwig uses in probability column names.

    Integral numbers lose their decimal point (1.0 -> "1") so numeric labels
    match the ``label_probabilities_0/1`` columns; booleans become "True"/"False".
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def probability_columns(columns: Sequence[str]) -> List[str]:
    return [
        str(col)
        for col in columns
        if str(col).startswith(PROBABILITIES_PREFIX)
    ]


def class_name_from_column(column: str) -> str:
    return column[len(PROBABILITIES_PREFIX):]


def _parse_vector(value: Any) -> Optional[List[float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        tokens = value.strip().strip("[]").replace(",", " ").split()
        try:
            return [float(tok) for tok in tokens]
        except ValueError:
            return None
    try:
        return [float(v) for v in list(value)]
    except (TypeError, ValueError):
        return None


def boolean_probability_labels(label_values: Sequence[Any]) -> Optional[Dict[str, str]]:
    """Which dataset label Ludwig's ``False``/``True`` probability columns stand for.

    Ludwig casts a numeric binary target to bool, so zero becomes False and any
    nonzero value True (for example labels 0/2 or 0/-1). Returns
    ``{"False": <zero label>, "True": <nonzero label>}`` as canonical strings,
    or None when the labels are not two numbers with exactly one zero.
    """
    numbers: Dict[str, float] = {}
    for value in label_values:
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if np.isfinite(number):
            numbers[canonical_label(value)] = number
    zeros = [name for name, number in numbers.items() if number == 0]
    if len(numbers) != 2 or len(zeros) != 1:
        return None
    nonzero = next(name for name in numbers if name != zeros[0])
    return {"False": zeros[0], "True": nonzero}


def normalize_prediction_frame(
    df_pred: pd.DataFrame,
    label_values: Sequence[Any],
) -> pd.DataFrame:
    """Reduce a Ludwig prediction table to the columns the ensemble averages.

    Per-class probability columns are kept (``label_probabilities_<class>``);
    a single vector column is expanded when Ludwig did not expand it. For
    numeric binary targets Ludwig names the columns after booleans, so they
    are renamed to the dataset's own label values (see
    ``boolean_probability_labels``).
    """
    out = df_pred.reset_index(drop=True).copy()
    prob_cols = probability_columns(out.columns)
    labels_as_str = sorted({canonical_label(v) for v in label_values})

    if not prob_cols and PROBABILITIES_VECTOR_COLUMN in out.columns:
        vectors = out[PROBABILITIES_VECTOR_COLUMN].map(_parse_vector)
        widths = {len(v) for v in vectors if v is not None}
        if len(widths) == 1:
            width = widths.pop()
            names = labels_as_str if len(labels_as_str) == width else [str(i) for i in range(width)]
            for idx, name in enumerate(names):
                out[f"{PROBABILITIES_PREFIX}{name}"] = vectors.map(
                    lambda v, i=idx: v[i] if v is not None else np.nan
                )
            prob_cols = probability_columns(out.columns)

    suffixes = {class_name_from_column(c) for c in prob_cols}
    bool_labels = boolean_probability_labels(label_values) if suffixes == {"False", "True"} else None
    if bool_labels:
        out = out.rename(
            columns={
                f"{PROBABILITIES_PREFIX}{flag}": f"{PROBABILITIES_PREFIX}{label}"
                for flag, label in bool_labels.items()
            }
        )
        prob_cols = probability_columns(out.columns)

    keep = [c for c in [PREDICTIONS_COLUMN, PROBABILITY_COLUMN] if c in out.columns] + prob_cols
    return out[keep]


def aggregate_member_predictions(
    frames: Sequence[pd.DataFrame],
    task_type: str,
) -> pd.DataFrame:
    """Average member predictions row-wise by their shared (base-row) index.

    Classification members contribute per-class probabilities; the averaged
    prediction is the arg-max class. Regression members contribute predicted
    values. ``n_models`` records how many members scored each row.
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    if task_type == "regression":
        value_cols = [PREDICTIONS_COLUMN]
    else:
        value_cols = probability_columns(frames[0].columns)
        if not value_cols:
            raise ValueError("No per-class probability columns found in member predictions.")

    stacked = pd.concat(
        [f[value_cols].apply(pd.to_numeric, errors="coerce") for f in frames]
    )
    grouped = stacked.groupby(level=0)
    averaged = grouped.mean()
    n_models = grouped.size().astype(int)

    if task_type == "regression":
        out = averaged[[PREDICTIONS_COLUMN]].copy()
    else:
        class_names = [class_name_from_column(c) for c in value_cols]
        probs = averaged[value_cols].to_numpy(dtype=float)
        best = probs.argmax(axis=1)
        out = pd.DataFrame(index=averaged.index)
        out[PREDICTIONS_COLUMN] = [class_names[i] for i in best]
        out[PROBABILITY_COLUMN] = probs.max(axis=1)
        for col in value_cols:
            out[col] = averaged[col]
    out[N_MODELS_COLUMN] = n_models
    return out.sort_index()


def apply_binary_threshold(
    aggregated: pd.DataFrame,
    thresholds: Any,
) -> pd.DataFrame:
    """Re-derive binary labels and confidence from a scalar or per-row decision threshold.

    ``thresholds`` is one float for every row or an array aligned with the
    rows of ``aggregated``. The threshold used for each row is recorded in a
    ``decision_threshold`` column so exported predictions stay auditable.
    """
    out = aggregated.copy()
    prob_cols = probability_columns(out.columns)
    names = [class_name_from_column(col) for col in prob_cols]
    if len(names) != 2 or out.empty:
        return out
    pos = positive_class_index(names)
    pos_scores = out[prob_cols[pos]].to_numpy(dtype=float)
    neg_scores = out[prob_cols[1 - pos]].to_numpy(dtype=float)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), pos_scores.shape)
    is_positive = pos_scores >= thresholds
    out[PREDICTIONS_COLUMN] = np.where(is_positive, names[pos], names[1 - pos])
    out[PROBABILITY_COLUMN] = np.where(is_positive, pos_scores, neg_scores)
    out[DECISION_THRESHOLD_COLUMN] = thresholds
    return out


def attach_row_metadata(
    aggregated: pd.DataFrame,
    base_df: pd.DataFrame,
    split_value: int,
    columns: Sequence[str],
    canonical_labels: bool = False,
) -> pd.DataFrame:
    """Prefix aggregated predictions with split/label/image columns from the base data.

    With ``canonical_labels`` the label column is written as the strings used in
    the probability column names, so downstream readers can match them.
    """
    out = aggregated.copy()
    insert_at = 0
    out.insert(insert_at, SPLIT_COLUMN_NAME, int(split_value))
    insert_at += 1
    for col in columns:
        if col in base_df.columns and col not in out.columns:
            values = base_df.loc[out.index, col]
            if canonical_labels and col == LABEL_COLUMN_NAME:
                values = values.map(canonical_label)
            out.insert(insert_at, col, values.to_numpy())
            insert_at += 1
    return out.reset_index(drop=True)


def positive_class_index(class_names: Sequence[str]) -> int:
    """Pick the positive class the same way the report picks its probability column.

    Two numeric classes with exactly one zero follow Ludwig (the nonzero label
    is positive); otherwise a name hint such as "true" or "1" decides, then the
    last name in sorted order.
    """
    nonzero = nonzero_binary_label(str(c) for c in class_names)
    if nonzero is not None:
        return [str(c) for c in class_names].index(nonzero)
    ordered = sorted(str(c) for c in class_names)
    chosen = ordered[-1]
    for name in ordered:
        if any(hint in name.lower() for hint in _POSITIVE_CLASS_HINTS):
            chosen = name
            break
    return [str(c) for c in class_names].index(chosen)


def _safe(fn, *args, **kwargs) -> Optional[float]:
    try:
        value = fn(*args, **kwargs)
    except Exception:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def compute_classification_statistics(
    y_true: Sequence[Any],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 3,
    threshold: Any = None,
) -> Dict[str, Any]:
    """Ludwig-shaped ``label`` statistics for averaged classification probabilities.

    Mirrors what Ludwig writes to test_statistics.json so the existing report
    code can render the ensemble like a single model: for two classes
    ``accuracy`` is sample accuracy; for more classes it is macro recall and
    ``accuracy_micro`` is sample accuracy. ``threshold`` (binary only) is one
    float or an array with one threshold per row.
    """
    class_names = [str(c) for c in class_names]
    y_true = np.asarray([canonical_label(v) for v in y_true])
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = len(class_names)
    binary = n_classes == 2

    if binary and threshold is not None:
        pos = positive_class_index(class_names)
        thresholds = np.broadcast_to(np.asarray(threshold, dtype=float), (len(y_true),))
        pred_idx = np.where(probabilities[:, pos] >= thresholds, pos, 1 - pos)
    else:
        pred_idx = probabilities.argmax(axis=1)
    y_pred = np.asarray(class_names)[pred_idx]

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    total = int(cm.sum())
    stats: Dict[str, Any] = {}

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den else 0.0
        per_class[name] = {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "accuracy": (tp + tn) / total if total else 0.0,
            "matthews_correlation_coefficient": float(mcc),
        }

    if binary:
        pos = positive_class_index(class_names)
        y_true_bin = (y_true == class_names[pos]).astype(int)
        y_pred_bin = (y_pred == class_names[pos]).astype(int)
        pos_scores = probabilities[:, pos]
        stats["accuracy"] = _safe(accuracy_score, y_true_bin, y_pred_bin)
        stats["balanced_accuracy"] = _safe(balanced_accuracy_score, y_true_bin, y_pred_bin)
        stats["precision"] = _safe(precision_score, y_true_bin, y_pred_bin, zero_division=0)
        stats["recall"] = _safe(recall_score, y_true_bin, y_pred_bin, zero_division=0)
        stats["specificity"] = per_class[class_names[pos]]["specificity"]
        stats["f1"] = _safe(f1_score, y_true_bin, y_pred_bin, zero_division=0)
        stats["mcc"] = _safe(matthews_corrcoef, y_true_bin, y_pred_bin)
        stats["roc_auc"] = _safe(roc_auc_score, y_true_bin, pos_scores)
        if len(np.unique(y_true_bin)) == 2:
            fpr, tpr, _ = roc_curve(y_true_bin, pos_scores)
            precisions, recalls, pr_thresholds = precision_recall_curve(y_true_bin, pos_scores)
            stats["roc_curve"] = {
                "false_positive_rate": fpr.tolist(),
                "true_positive_rate": tpr.tolist(),
            }
            stats["precision_recall_curve"] = {
                "precisions": precisions.tolist(),
                "recalls": recalls.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
    else:
        stats["accuracy"] = _safe(balanced_accuracy_score, y_true, y_pred)
        stats["accuracy_micro"] = _safe(accuracy_score, y_true, y_pred)
        k = max(1, min(int(top_k), n_classes))
        y_true_idx = np.array([class_names.index(v) if v in class_names else -1 for v in y_true])
        if (y_true_idx >= 0).all():
            stats["hits_at_k"] = _safe(
                top_k_accuracy_score, y_true_idx, probabilities, k=k, labels=list(range(n_classes))
            )
            if len(np.unique(y_true_idx)) == n_classes:
                stats["roc_auc"] = _safe(
                    roc_auc_score, y_true_idx, probabilities, multi_class="ovr", average="macro",
                    labels=list(range(n_classes)),
                )
        stats["overall_stats"] = {
            "avg_precision_macro": _safe(precision_score, y_true, y_pred, average="macro", zero_division=0),
            "avg_recall_macro": _safe(recall_score, y_true, y_pred, average="macro", zero_division=0),
            "avg_f1_score_macro": _safe(f1_score, y_true, y_pred, average="macro", zero_division=0),
            "avg_precision_micro": _safe(precision_score, y_true, y_pred, average="micro", zero_division=0),
            "avg_recall_micro": _safe(recall_score, y_true, y_pred, average="micro", zero_division=0),
            "avg_f1_score_micro": _safe(f1_score, y_true, y_pred, average="micro", zero_division=0),
            "avg_f1_score_weighted": _safe(f1_score, y_true, y_pred, average="weighted", zero_division=0),
            "kappa_score": _safe(cohen_kappa_score, y_true, y_pred),
        }

    stats = {k: v for k, v in stats.items() if v is not None}
    stats["per_class_stats"] = per_class
    stats["confusion_matrix"] = cm.tolist()
    return stats


def compute_regression_statistics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> Dict[str, Any]:
    """Ludwig-shaped ``label`` statistics for averaged regression predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {}
    mse = float(mean_squared_error(y_true, y_pred))
    stats: Dict[str, Any] = {
        "loss": mse,
        "mean_squared_error": mse,
        "root_mean_squared_error": float(np.sqrt(mse)),
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "r2": _safe(r2_score, y_true, y_pred),
    }
    nonzero = y_true != 0
    if nonzero.any():
        rel = (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
        stats["mean_absolute_percentage_error"] = float(np.mean(np.abs(rel)))
        stats["root_mean_squared_percentage_error"] = float(np.sqrt(np.mean(rel ** 2)))
    return {k: v for k, v in stats.items() if v is not None}


def flatten_label_statistics(label_stats: Dict[str, Any]) -> Dict[str, float]:
    """Scalar metrics from Ludwig-shaped label statistics, as the report tables want them."""
    output_type = detect_output_type({"label": label_stats})
    return extract_test_metrics({"label": label_stats}, output_type)


def member_metrics(
    train_stats: Optional[Dict[str, Any]],
    test_stats: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Last-epoch train/validation metrics and test metrics of one member."""
    if not test_stats:
        return {"training": {}, "validation": {}, "test": {}}
    output_type = detect_output_type(test_stats)
    metrics = extract_metrics_from_json(train_stats or {}, test_stats, output_type)
    cleaned: Dict[str, Dict[str, float]] = {}
    for split, values in metrics.items():
        cleaned[split] = {
            k: float(v)
            for k, v in (values or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        }
    return cleaned


def summarize_member_metrics(
    members: Sequence[Dict[str, Any]],
    split: str,
) -> Dict[str, Dict[str, float]]:
    """Mean and sample standard deviation per metric across trained members."""
    values: Dict[str, List[float]] = {}
    for member in members:
        for key, value in (member.get(split) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    summary: Dict[str, Dict[str, float]] = {}
    for key, series in values.items():
        summary[key] = {
            "mean": float(np.mean(series)),
            "std": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
            "n": len(series),
        }
    return summary
