from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve
)


# -------------------- Transparent Metrics (task-aware) -------------------- #

def _safe_y_proba_to_array(y_proba) -> Optional[np.ndarray]:
    """Convert predictor.predict_proba output (array/DataFrame/dict) to np.ndarray or None."""
    if y_proba is None:
        return None
    if isinstance(y_proba, pd.DataFrame):
        return y_proba.values
    if isinstance(y_proba, (list, tuple)):
        return np.asarray(y_proba)
    if isinstance(y_proba, np.ndarray):
        return y_proba
    if isinstance(y_proba, dict):
        try:
            return np.vstack([np.asarray(v) for _, v in sorted(y_proba.items())]).T
        except Exception:
            return None
    return None


def _specificity_from_cm(cm: np.ndarray) -> float:
    """Specificity (TNR) for binary confusion matrix."""
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    denom = (tn + fp)
    return float(tn / denom) if denom > 0 else np.nan


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> "OrderedDict[str, float]":
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    # Avoid division by zero using clip
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100.0)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    metrics = OrderedDict()
    metrics["MSE"] = mse
    metrics["RMSE"] = rmse
    metrics["MAE"] = mae
    metrics["MAPE_%"] = mape
    metrics["R2"] = r2
    metrics["MedianAE"] = medae
    return metrics


def _get_binary_scores(
    y_true: pd.Series,
    y_proba: Optional[np.ndarray],
    predictor,
) -> Tuple[np.ndarray, object, Optional[np.ndarray]]:
    classes_sorted = np.sort(pd.unique(y_true))
    pos_label = classes_sorted[-1]
    pos_scores = None
    if y_proba is not None:
        if y_proba.ndim == 1:
            pos_scores = y_proba
        else:
            pos_col_idx = -1
            try:
                if hasattr(predictor, "class_labels") and predictor.class_labels:
                    pos_col_idx = list(predictor.class_labels).index(pos_label)
            except Exception:
                pos_col_idx = -1
            pos_scores = y_proba[:, pos_col_idx]
    return classes_sorted, pos_label, pos_scores


def _compute_binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: Optional[np.ndarray],
    predictor,
    classes_sorted: Optional[np.ndarray] = None,
    pos_label: Optional[object] = None,
    pos_scores: Optional[np.ndarray] = None,
) -> "OrderedDict[str, float]":
    metrics = OrderedDict()
    if classes_sorted is None or pos_label is None or pos_scores is None:
        classes_sorted, pos_label, pos_scores = _get_binary_scores(y_true, y_proba, predictor)

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics["Recall_(Sensitivity/TPR)"] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics["F1-Score"] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    try:
        cm = confusion_matrix(y_true, y_pred, labels=classes_sorted)
        metrics["Specificity_(TNR)"] = _specificity_from_cm(cm)
    except Exception:
        metrics["Specificity_(TNR)"] = np.nan

    # Probabilistic metrics
    if y_proba is not None and pos_scores is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true == pos_label, pos_scores)
        except Exception:
            metrics["ROC-AUC"] = np.nan
        try:
            metrics["PR-AUC"] = average_precision_score(y_true == pos_label, pos_scores)
        except Exception:
            metrics["PR-AUC"] = np.nan
        try:
            if y_proba.ndim == 1:
                y_proba_ll = np.column_stack([1 - pos_scores, pos_scores])
            else:
                y_proba_ll = y_proba
            metrics["LogLoss"] = log_loss(y_true, y_proba_ll, labels=classes_sorted)
        except Exception:
            metrics["LogLoss"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan
        metrics["LogLoss"] = np.nan

    try:
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics["MCC"] = np.nan

    return metrics


def _compute_multiclass_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: Optional[np.ndarray]
) -> "OrderedDict[str, float]":
    metrics = OrderedDict()
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Macro Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Macro Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Macro F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Weighted Precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["Weighted Recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["Weighted F1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        metrics["Cohen_Kappa"] = cohen_kappa_score(y_true, y_pred)
    except Exception:
        metrics["Cohen_Kappa"] = np.nan
    try:
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics["MCC"] = np.nan

    # Probabilistic metrics
    classes_sorted = np.sort(pd.unique(y_true))
    if y_proba is not None and y_proba.ndim == 2:
        try:
            metrics["LogLoss"] = log_loss(y_true, y_proba, labels=classes_sorted)
        except Exception:
            metrics["LogLoss"] = np.nan
        # Macro ROC-AUC / PR-AUC via OVR
        try:
            class_to_index = {c: i for i, c in enumerate(classes_sorted)}
            y_true_idx = np.vectorize(class_to_index.get)(y_true)
            metrics["ROC-AUC_macro"] = roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="macro")
        except Exception:
            metrics["ROC-AUC_macro"] = np.nan
        try:
            Y_true_ind = np.zeros_like(y_proba)
            idx_map = {c: i for i, c in enumerate(classes_sorted)}
            Y_true_ind[np.arange(y_proba.shape[0]), np.vectorize(idx_map.get)(y_true)] = 1
            metrics["PR-AUC_macro"] = average_precision_score(Y_true_ind, y_proba, average="macro")
        except Exception:
            metrics["PR-AUC_macro"] = np.nan
    else:
        metrics["LogLoss"] = np.nan
        metrics["ROC-AUC_macro"] = np.nan
        metrics["PR-AUC_macro"] = np.nan

    return metrics


def aggregate_metrics(list_of_dicts):
    """Aggregate a list of metrics dicts (per split) into mean/std."""
    agg_mean = {}
    agg_std = {}
    for split in ("Train", "Validation", "Test", "Test (external)"):
        keys = set()
        for m in list_of_dicts:
            if isinstance(m, dict) and split in m:
                keys.update(m[split].keys())
        if not keys:
            continue
        agg_mean[split] = {}
        agg_std[split] = {}
        for k in keys:
            vals = [m[split][k] for m in list_of_dicts if split in m and k in m[split]]
            numeric_vals = []
            for v in vals:
                try:
                    numeric_vals.append(float(v))
                except Exception:
                    pass
            if numeric_vals:
                agg_mean[split][k] = float(np.mean(numeric_vals))
                agg_std[split][k] = float(np.std(numeric_vals, ddof=0))
            else:
                agg_mean[split][k] = vals[-1] if vals else None
                agg_std[split][k] = None
    return agg_mean, agg_std


def compute_metrics_for_split(
    predictor,
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    threshold: Optional[float] = None,    # <— NEW
    return_curve: bool = False,
) -> "OrderedDict[str, float] | Tuple[OrderedDict[str, float], Optional[dict]]":
    """Compute transparency metrics for one split (Train/Val/Test) based on task type."""
    # Prepare inputs
    features = df.drop(columns=[target_col], errors="ignore")
    y_true_series = df[target_col].reset_index(drop=True)

    # Probabilities (if available)
    y_proba = None
    try:
        y_proba_raw = predictor.predict_proba(features)
        y_proba = _safe_y_proba_to_array(y_proba_raw)
    except Exception:
        y_proba = None

    classes_sorted = pos_label = pos_scores = None
    if problem_type == "binary":
        classes_sorted, pos_label, pos_scores = _get_binary_scores(y_true_series, y_proba, predictor)

    # Labels (optionally thresholded for binary)
    y_pred_series = None
    if problem_type == "binary" and (threshold is not None) and (pos_scores is not None):
        neg_label = classes_sorted[0]
        y_pred_series = pd.Series(np.where(pos_scores >= float(threshold), pos_label, neg_label)).reset_index(drop=True)
    else:
        # Fall back to model's default label prediction (argmax / 0.5 equivalent)
        y_pred_series = pd.Series(predictor.predict(features)).reset_index(drop=True)

    if problem_type == "regression":
        y_true_arr = np.asarray(y_true_series, dtype=float)
        y_pred_arr = np.asarray(y_pred_series, dtype=float)
        metrics = _compute_regression_metrics(y_true_arr, y_pred_arr)
        return (metrics, None) if return_curve else metrics

    if problem_type == "binary":
        metrics = _compute_binary_metrics(
            y_true_series,
            y_pred_series,
            y_proba,
            predictor,
            classes_sorted=classes_sorted,
            pos_label=pos_label,
            pos_scores=pos_scores,
        )
        roc_curve_data = None
        if return_curve and pos_scores is not None and pos_label is not None:
            try:
                fpr, tpr, thresholds = roc_curve(y_true_series == pos_label, pos_scores)
                roc_curve_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }
            except Exception:
                roc_curve_data = None
        return (metrics, roc_curve_data) if return_curve else metrics

    # multiclass
    metrics = _compute_multiclass_metrics(y_true_series, y_pred_series, y_proba)
    return (metrics, None) if return_curve else metrics


def evaluate_all_transparency(
    predictor,
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    target_col: str,
    problem_type: str,
    threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, dict]]:
    """
    Evaluate Train/Val/Test with the transparent metrics suite.
    Returns:
      - metrics_table: DataFrame with index=Metric, columns subset of [Train, Validation, Test]
      - raw_dict: nested dict {split -> {metric -> value}}
      - roc_curves: nested dict {split -> {fpr, tpr, thresholds}} (binary only)
    """
    split_results: Dict[str, Dict[str, float]] = {}
    roc_curves: Dict[str, dict] = {}
    splits = []

    # IMPORTANT: do NOT apply threshold to Train/Val
    if train_df is not None and len(train_df):
        train_metrics, train_curve = compute_metrics_for_split(
            predictor,
            train_df,
            target_col,
            problem_type,
            threshold=None,
            return_curve=True,
        )
        split_results["Train"] = train_metrics
        if train_curve:
            roc_curves["Train"] = train_curve
        splits.append("Train")
    if val_df is not None and len(val_df):
        val_metrics, val_curve = compute_metrics_for_split(
            predictor,
            val_df,
            target_col,
            problem_type,
            threshold=None,
            return_curve=True,
        )
        split_results["Validation"] = val_metrics
        if val_curve:
            roc_curves["Validation"] = val_curve
        splits.append("Validation")
    if test_df is not None and len(test_df):
        test_metrics, test_curve = compute_metrics_for_split(
            predictor,
            test_df,
            target_col,
            problem_type,
            threshold=threshold,
            return_curve=True,
        )
        split_results["Test"] = test_metrics
        if test_curve:
            roc_curves["Test"] = test_curve
        splits.append("Test")

    # Preserve order from the first split; include any extras from others
    order_source = split_results[splits[0]] if splits else {}
    all_metrics = list(order_source.keys())
    for s in splits[1:]:
        for m in split_results[s].keys():
            if m not in all_metrics:
                all_metrics.append(m)

    metrics_table = pd.DataFrame(index=all_metrics, columns=splits, dtype=float)
    for s in splits:
        for m, v in split_results[s].items():
            metrics_table.loc[m, s] = v

    return metrics_table, split_results, roc_curves
