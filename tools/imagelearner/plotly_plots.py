import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from calibration_plot import expected_calibration_error
from constants import DEFAULT_HITS_AT_K, LABEL_COLUMN_NAME, SPLIT_COLUMN_NAME
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _style_fig(fig: go.Figure, font_size: int = 12) -> go.Figure:
    """Apply consistent styling across Plotly figures."""
    fig.update_layout(
        font=dict(size=font_size),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(gridcolor="#e8e8e8")
    fig.update_yaxes(gridcolor="#e8e8e8")
    return fig


def _fig_to_html(
    fig: go.Figure, *, include_js: bool = False, config: Optional[dict] = None
) -> str:
    """Render a Plotly figure to a lightweight HTML fragment."""
    include_plotlyjs = "cdn" if include_js else False
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config=config,
    )


def _wrap_plot(
    title: str,
    fig: go.Figure,
    *,
    include_js: bool = False,
    config: Optional[dict] = None,
) -> Dict[str, str]:
    """Package a figure with its title for downstream HTML rendering."""
    return {"title": title, "html": _fig_to_html(fig, include_js=include_js, config=config)}


def _strip_prob_prefix(col: str) -> str:
    if col.startswith("label_probabilities_"):
        return col.replace("label_probabilities_", "")
    if col.startswith("probabilities_"):
        return col.replace("probabilities_", "")
    return col


def _select_positive_probability_column(
    prob_cols: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    if not prob_cols:
        return None, None
    prob_cols_sorted = sorted(prob_cols)
    preferred_keys = ("event", "true", "positive", "pos", "1")
    for col in prob_cols_sorted:
        suffix = _strip_prob_prefix(col)
        if any(k in suffix.lower() for k in preferred_keys):
            return col, suffix
    return prob_cols_sorted[-1], _strip_prob_prefix(prob_cols_sorted[-1])


def _resolve_positive_label_from_probability(
    labels: np.ndarray,
    positive_label_hint: Optional[str],
) -> Optional[object]:
    unique_labels = list(pd.unique(labels))
    if not unique_labels:
        return None

    if positive_label_hint is not None:
        hint = str(positive_label_hint)
        for label in unique_labels:
            if str(label) == hint:
                return label

    if len(unique_labels) == 2:
        return unique_labels[1]
    return unique_labels[0]


def _line_chart(
    traces: List[tuple],
    *,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Build a basic epoch-indexed line chart for train/val/test curves."""
    fig = go.Figure()
    for name, series in traces:
        if not series:
            continue
        epochs = list(range(1, len(series) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=series,
                mode="lines+markers",
                name=name,
                line=dict(width=4),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Epoch",
        yaxis_title=yaxis_title,
        width=760,
        height=520,
        hovermode="x unified",
    )
    _style_fig(fig)
    return fig


def _labels_from_metadata_dict(meta_dict: dict) -> List[str]:
    """Extract ordered label names from Ludwig train_set_metadata."""
    if not isinstance(meta_dict, dict):
        return []

    for key in ("idx2str", "idx2label", "vocab"):
        seq = meta_dict.get(key)
        if isinstance(seq, list) and seq:
            return [str(v) for v in seq]

    str2idx = meta_dict.get("str2idx")
    if isinstance(str2idx, dict) and str2idx:
        int_indices = [v for v in str2idx.values() if isinstance(v, int)]
        if int_indices:
            max_idx = max(int_indices)
            ordered = [None] * (max_idx + 1)
            for name, idx in str2idx.items():
                if isinstance(idx, int) and 0 <= idx < len(ordered):
                    ordered[idx] = name
            return [str(v) for v in ordered if v is not None]

    return []


def _resolve_confusion_labels(
    label_stats: dict,
    n_classes: int,
    metadata_csv_path: Optional[str],
    train_set_metadata_path: Optional[str],
) -> List[str]:
    """Prefer original labels from metadata; fall back to stats if unavailable."""
    if train_set_metadata_path:
        try:
            meta_path = Path(train_set_metadata_path)
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta_json = json.load(f)
                label_meta = meta_json.get(LABEL_COLUMN_NAME)
                if not isinstance(label_meta, dict):
                    label_meta = next(
                        (
                            v
                            for v in meta_json.values()
                            if isinstance(v, dict)
                            and any(k in v for k in ("idx2str", "str2idx", "idx2label", "vocab"))
                        ),
                        None,
                    )
                labels_from_meta = _labels_from_metadata_dict(label_meta) if label_meta else []
                if labels_from_meta and len(labels_from_meta) >= n_classes:
                    return [str(label) for label in labels_from_meta[:n_classes]]
        except Exception as exc:
            print(f"Warning: Unable to read labels from train_set_metadata: {exc}")

    if metadata_csv_path:
        try:
            csv_path = Path(metadata_csv_path)
            if csv_path.exists():
                df_meta = pd.read_csv(csv_path)
                if LABEL_COLUMN_NAME in df_meta.columns:
                    uniques = df_meta[LABEL_COLUMN_NAME].dropna().unique().tolist()
                    if uniques and len(uniques) >= n_classes:
                        return [str(u) for u in uniques[:n_classes]]
        except Exception as exc:
            print(f"Warning: Unable to read labels from metadata CSV: {exc}")

    pcs = label_stats.get("per_class_stats", {})
    if pcs:
        pcs_labels = [str(k) for k in pcs.keys()]
        if len(pcs_labels) >= n_classes:
            return pcs_labels[:n_classes]

    labels = label_stats.get("labels")
    if not labels:
        labels = [str(i) for i in range(n_classes)]
    if len(labels) < n_classes:
        labels = labels + [str(i) for i in range(len(labels), n_classes)]
    return [str(label) for label in labels[:n_classes]]


def _class_index(raw_label) -> Optional[int]:
    """Return a class index for integer-like labels, otherwise None."""
    if isinstance(raw_label, bool):
        return None
    if isinstance(raw_label, int):
        return raw_label
    if isinstance(raw_label, float):
        return int(raw_label) if raw_label.is_integer() else None

    raw_str = str(raw_label).strip()
    if not raw_str:
        return None
    try:
        idx_val = float(raw_str)
    except Exception:
        return None
    if idx_val.is_integer():
        return int(idx_val)
    return None


def _resolve_display_labels(raw_labels: List, friendly_labels: Optional[List[str]]) -> List[str]:
    """Map encoded class ids like 0/1 back to friendly labels when available."""
    if not friendly_labels:
        return [str(label) for label in raw_labels]

    friendly_labels = [str(label) for label in friendly_labels]
    raw_indices = [_class_index(raw_label) for raw_label in raw_labels]
    if (
        raw_indices
        and all(idx is not None for idx in raw_indices)
        and set(raw_indices) == set(range(len(raw_indices)))
        and max(raw_indices) < len(friendly_labels)
    ):
        return [friendly_labels[idx] for idx in raw_indices]

    friendly_lookup = set(friendly_labels)
    display_labels: List[str] = []
    for raw_label, idx in zip(raw_labels, raw_indices):
        raw_str = str(raw_label)
        if raw_str in friendly_lookup:
            display_labels.append(raw_str)
            continue

        if idx is not None and 0 <= idx < len(friendly_labels):
            display_labels.append(friendly_labels[idx])
        else:
            display_labels.append(raw_str)

    return display_labels


def build_classification_plots(
    test_stats_path: str,
    training_stats_path: Optional[str] = None,
    metadata_csv_path: Optional[str] = None,
    train_set_metadata_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, str]]:
    """
    Read Ludwig’s test_statistics.json and build three interactive Plotly panels:
      - Confusion Matrix
      - ROC-AUC
      - Classification Report Heatmap

    If metadata paths are provided, the confusion matrix axes will use the original
    label values from the training metadata rather than integer-encoded labels.

    Returns a list of dicts, each with:
      {
        "title": <plot title>,
        "html":  <HTML fragment for embedding>
      }
    """
    # --- Load test stats ---
    with open(test_stats_path, "r") as f:
        test_stats = json.load(f)
    label_stats = test_stats["label"]

    # common sizing
    cell = 40
    n_classes = len(label_stats["confusion_matrix"])
    side_px = max(cell * n_classes + 200, 600)
    common_cfg = {"displayModeBar": True, "scrollZoom": True}

    plots: List[Dict[str, str]] = []

    # 0) Confusion Matrix
    cm = np.array(label_stats["confusion_matrix"], dtype=int)
    labels = _resolve_confusion_labels(
        label_stats,
        n_classes,
        metadata_csv_path=metadata_csv_path,
        train_set_metadata_path=train_set_metadata_path,
    )
    total = cm.sum()

    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig_cm.update_traces(xgap=2, ygap=2)
    cm_title = "Confusion Matrix"
    if threshold is not None:
        cm_title = f"Confusion Matrix (Threshold: {threshold})"
    fig_cm.update_layout(
        title=dict(text=cm_title, x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Observed",
        yaxis_autorange="reversed",
        width=side_px,
        height=side_px,
        margin=dict(t=100, l=80, r=80, b=80),
    )
    _style_fig(fig_cm)

    # annotate counts and percentages
    mval = cm.max() if cm.size else 0
    thresh = mval / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            pct = (v / total * 100) if total > 0 else 0
            color = "white" if v > thresh else "black"
            fig_cm.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"<b>{v}</b>",
                showarrow=False,
                font=dict(color=color, size=14),
                xanchor="center",
                yanchor="bottom",
                yshift=2,
            )
            fig_cm.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{pct:.1f}%",
                showarrow=False,
                font=dict(color=color, size=13),
                xanchor="center",
                yanchor="top",
                yshift=-2,
            )

    plots.append(
        _wrap_plot("Confusion Matrix", fig_cm, include_js=True, config=common_cfg)
    )

    # 1) ROC / PR curves only for binary tasks
    if n_classes == 2:
        roc_plot = _build_static_roc_plot(label_stats, common_cfg, friendly_labels=labels)
        if roc_plot:
            plots.append(roc_plot)

        pr_plot = _build_precision_recall_plot(label_stats, common_cfg)
        if pr_plot:
            plots.append(pr_plot)

    # 2) Classification Report Heatmap
    pcs = label_stats.get("per_class_stats", {})
    if pcs:
        classes = list(pcs.keys())
        display_classes = _resolve_display_labels(classes, labels)
        metrics = [
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "matthews_correlation_coefficient",
            "specificity",
        ]
        z, txt = [], []
        for c in classes:
            row, trow = [], []
            for m in metrics:
                val = pcs[c].get(m, 0)
                row.append(val)
                trow.append(f"{val:.2f}")
            z.append(row)
            txt.append(trow)

        fig_cr = go.Figure(
            go.Heatmap(
                z=z,
                x=[m.replace("_", " ") for m in metrics],
                y=display_classes,
                text=txt,
                texttemplate="%{text}",
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Value"),
            )
        )
        fig_cr.update_layout(
            title="Per-Class metrics",
            xaxis_title="",
            yaxis_title="Class",
            width=side_px,
            height=side_px,
            margin=dict(t=80, l=80, r=80, b=80),
        )
        _style_fig(fig_cr)
        plots.append(
            _wrap_plot("Per-Class metrics", fig_cr, config=common_cfg)
        )

    # 3) Prediction Diagnostics (from predictions.csv)
    # Note: appended separately in generate_html_report, not returned here.

    return plots


def build_train_validation_plots(
    train_stats_path: str, top_k: int = DEFAULT_HITS_AT_K
) -> List[Dict[str, str]]:
    """Generate Train/Validation learning curve plots from training_statistics.json."""
    if not train_stats_path or not Path(train_stats_path).exists():
        return []
    try:
        with open(train_stats_path, "r") as f:
            train_stats = json.load(f)
    except Exception as exc:
        print(f"Warning: Unable to read training statistics: {exc}")
        return []

    label_train = (train_stats.get("training") or {}).get("label", {})
    label_val = (train_stats.get("validation") or {}).get("label", {})
    if not label_train and not label_val:
        return []
    plots: List[Dict[str, str]] = []
    include_js = True  # Load Plotly.js once for this group

    def _get_series(stats: dict, metric: str) -> List[float]:
        vals = stats.get(metric, [])
        if isinstance(vals, list):
            return [float(v) for v in vals]
        try:
            return [float(vals)]
        except Exception:
            return []

    metric_specs = [
        ("loss", "Loss across epochs", "Loss"),
        ("accuracy", "Accuracy across epochs", "Accuracy"),
        (
            "hits_at_k",
            f"Hits@{int(top_k)} across epochs (correct class in top {int(top_k)})",
            f"Hits@{int(top_k)}",
        ),
        ("roc_auc", "ROC-AUC across epochs", "ROC-AUC"),
        ("precision", "Precision across epochs", "Precision"),
        ("recall", "Recall/Sensitivity across epochs", "Recall"),
        ("specificity", "Specificity across epochs", "Specificity"),
    ]

    for key, title, yaxis in metric_specs:
        train_series = _get_series(label_train, key)
        val_series = _get_series(label_val, key)
        if not train_series and not val_series:
            continue
        fig = _line_chart(
            [("Train", train_series), ("Validation", val_series)],
            title=title,
            yaxis_title=yaxis,
        )
        plots.append(_wrap_plot(title, fig, include_js=include_js))
        include_js = False

    # Precision vs Recall evolution (validation)
    val_prec = _get_series(label_val, "precision")
    val_rec = _get_series(label_val, "recall")
    if val_prec and val_rec:
        max_len = min(len(val_prec), len(val_rec))
        fig_pr = _line_chart(
            [
                ("Precision", val_prec[:max_len]),
                ("Recall", val_rec[:max_len]),
            ],
            title="Validation Precision and Recall by Epoch",
            yaxis_title="Value",
        )
        plots.append(_wrap_plot("Precision vs Recall Evolution", fig_pr, include_js=include_js))
        include_js = False

    def _compute_f1(p: List[float], r: List[float]) -> List[float]:
        return [
            0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            for prec, rec in zip(p, r)
        ]

    f1_train = _compute_f1(_get_series(label_train, "precision"), _get_series(label_train, "recall"))
    f1_val = _compute_f1(val_prec, val_rec)
    if f1_train or f1_val:
        fig_f1 = _line_chart(
            [("Train", f1_train), ("Validation", f1_val)],
            title="F1-Score across epochs (derived)",
            yaxis_title="F1-Score",
        )
        plots.append(_wrap_plot("F1-Score across epochs (derived)", fig_f1, include_js=include_js))
        include_js = False

    # Overfitting Gap: Train vs Val ROC-AUC (gap)
    roc_train = _get_series(label_train, "roc_auc")
    roc_val = _get_series(label_val, "roc_auc")
    if roc_train and roc_val:
        max_len = min(len(roc_train), len(roc_val))
        gaps = [t - v for t, v in zip(roc_train[:max_len], roc_val[:max_len])]
        fig_gap = _line_chart(
            [("Train - Val ROC-AUC", gaps)],
            title="Overfitting gap: ROC-AUC across epochs",
            yaxis_title="Gap",
        )
        plots.append(_wrap_plot("Overfitting gap: ROC-AUC across epochs", fig_gap, include_js=include_js))
        include_js = False

    # Best Epoch Dashboard (based on max val ROC-AUC)
    if roc_val:
        best_idx = int(np.argmax(roc_val))
        best_epoch = best_idx + 1
        metrics_at_best: Dict[str, Optional[float]] = {
            "ROC-AUC": roc_val[best_idx] if best_idx < len(roc_val) else None
        }

        for metric_key, label in [
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("hits_at_k", f"Hits@{int(top_k)}"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("specificity", "Specificity"),
            ("loss", "Loss"),
        ]:
            series = _get_series(label_val, metric_key)
            if series and best_idx < len(series):
                metrics_at_best[label] = series[best_idx]

        if f1_val and best_idx < len(f1_val):
            metrics_at_best["F1-Score (derived)"] = f1_val[best_idx]

        fig_best = go.Figure()
        for name, value in metrics_at_best.items():
            if value is not None:
                fig_best.add_trace(go.Bar(name=name, x=[name], y=[value]))
        fig_best.update_layout(
            title=dict(text=f"Best Epoch Dashboard (Val ROC-AUC @ epoch {best_epoch})", x=0.5),
            xaxis_title="Metric",
            yaxis_title="Value",
            width=760,
            height=520,
            showlegend=False,
        )
        _style_fig(fig_best)
        plots.append(_wrap_plot("Best Validation Epoch Snapshot (Metrics)", fig_best, include_js=include_js))

    return plots


def _get_regression_series(split_stats: dict, metric: str) -> List[float]:
    if metric not in split_stats:
        return []
    vals = split_stats.get(metric, [])
    if isinstance(vals, list):
        return [float(v) for v in vals]
    try:
        return [float(vals)]
    except Exception:
        return []


def _regression_line_plot(
    train_split: dict,
    val_split: dict,
    metric_key: str,
    title: str,
    yaxis_title: str,
    include_js: bool,
) -> Optional[Dict[str, str]]:
    train_series = _get_regression_series(train_split, metric_key)
    val_series = _get_regression_series(val_split, metric_key)
    if not train_series and not val_series:
        return None

    fig = _line_chart(
        [("Train", train_series), ("Validation", val_series)],
        title=title,
        yaxis_title=yaxis_title,
    )
    return _wrap_plot(title, fig, include_js=include_js)


def build_regression_train_val_plots(train_stats_path: str) -> List[Dict[str, str]]:
    """Generate regression Train/Validation learning curve plots from training_statistics.json."""
    if not train_stats_path or not Path(train_stats_path).exists():
        return []
    try:
        with open(train_stats_path, "r") as f:
            train_stats = json.load(f)
    except Exception as exc:
        print(f"Warning: Unable to read training statistics: {exc}")
        return []

    label_train = (train_stats.get("training") or {}).get("label", {})
    label_val = (train_stats.get("validation") or {}).get("label", {})
    if not label_train and not label_val:
        return []

    plots: List[Dict[str, str]] = []
    include_js = True
    for metric_key, title, ytitle in [
        ("mean_absolute_error", "Mean Absolute Error across epochs", "MAE"),
        ("root_mean_squared_error", "Root Mean Squared Error across epochs", "RMSE"),
        ("mean_absolute_percentage_error", "Mean Absolute Percentage Error across epochs", "MAPE"),
        ("r2", "R² across epochs", "R²"),
        ("loss", "Loss across epochs", "Loss"),
    ]:
        plot = _regression_line_plot(label_train, label_val, metric_key, title, ytitle, include_js)
        if plot:
            plots.append(plot)
            include_js = False
    return plots


def build_regression_test_plots(train_stats_path: str) -> List[Dict[str, str]]:
    """Generate regression Test learning curves from training_statistics.json."""
    if not train_stats_path or not Path(train_stats_path).exists():
        return []
    try:
        with open(train_stats_path, "r") as f:
            train_stats = json.load(f)
    except Exception as exc:
        print(f"Warning: Unable to read training statistics: {exc}")
        return []

    label_test = (train_stats.get("test") or {}).get("label", {})
    if not label_test:
        return []

    plots: List[Dict[str, str]] = []
    include_js = True
    metrics = [
        ("mean_absolute_error", "Mean Absolute Error Across Epochs", "MAE"),
        ("root_mean_squared_error", "Root Mean Squared Error Across Epochs", "RMSE"),
        ("mean_absolute_percentage_error", "Mean Absolute Percentage Error Across Epochs", "MAPE"),
        ("r2", "R² Across Epochs", "R²"),
        ("loss", "Loss Across Epochs", "Loss"),
    ]
    for metric_key, title, ytitle in metrics:
        series = _get_regression_series(label_test, metric_key)
        if not series:
            continue
        fig = _line_chart(
            [("Test", series)],
            title=title,
            yaxis_title=ytitle,
        )
        plots.append(_wrap_plot(title, fig, include_js=include_js))
        include_js = False
    return plots


def _build_static_roc_plot(
    label_stats: dict,
    config: dict,
    friendly_labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
) -> Optional[Dict[str, str]]:
    """Build ROC curve directly from test_statistics.json (single curve)."""
    roc_data = label_stats.get("roc_curve")
    if not isinstance(roc_data, dict):
        return None

    fpr = roc_data.get("false_positive_rate")
    tpr = roc_data.get("true_positive_rate")
    if not fpr or not tpr or len(fpr) != len(tpr):
        return None

    try:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines+markers",
                name="ROC Curve",
                line=dict(color="#1f77b4", width=4),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=2, dash="dash"),
                hovertemplate="Random Classifier<extra></extra>",
            )
        )

        auc_val = label_stats.get("roc_auc") or label_stats.get("roc_auc_macro") or label_stats.get("roc_auc_micro")
        auc_txt = f" (AUC = {auc_val:.3f})" if isinstance(auc_val, (int, float)) else ""

        # Determine which label is treated as positive for the curve
        label_list: List = []
        pcs = label_stats.get("per_class_stats", {})
        if pcs:
            label_list = list(pcs.keys())
        if not label_list:
            labels_from_stats = label_stats.get("labels")
            if isinstance(labels_from_stats, list):
                label_list = labels_from_stats

        # Try to resolve index of the positive label explicitly provided by Ludwig
        pos_label_raw = (
            roc_data.get("positive_label")
            or roc_data.get("positive_class")
            or label_stats.get("positive_label")
        )
        pos_label_idx = None
        if pos_label_raw is not None and isinstance(label_list, list):
            try:
                pos_label_idx = label_list.index(pos_label_raw)
            except ValueError:
                pos_label_idx = None

        # Fallback: use the second label if available, otherwise the first
        if pos_label_idx is None:
            if isinstance(label_list, list) and len(label_list) >= 2:
                pos_label_idx = 1
            elif isinstance(label_list, list) and label_list:
                pos_label_idx = 0

        if pos_label_raw is None and isinstance(label_list, list) and pos_label_idx is not None:
            pos_label_raw = label_list[pos_label_idx]

        # Map to friendly label if we have one from metadata/CSV
        pos_label_display = pos_label_raw
        if (
            friendly_labels
            and isinstance(pos_label_idx, int)
            and 0 <= pos_label_idx < len(friendly_labels)
        ):
            pos_label_display = friendly_labels[pos_label_idx]

        pos_label_txt = (
            f"Positive class: {pos_label_display}"
            if pos_label_display is not None
            else "Positive class: (not available)"
        )

        title_label = f"ROC Curve{auc_txt}"
        if pos_label_display is not None:
            title_label = f"ROC Curve (Positive Class: {pos_label_display}){auc_txt}"

        fig.update_layout(
            title=dict(text=title_label, x=0.5),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=700,
            height=600,
            margin=dict(t=80, l=80, r=80, b=110),
            hovermode="closest",
            legend=dict(
                x=0.6,
                y=0.1,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )
        _style_fig(fig)
        fig.update_xaxes(range=[0, 1.0])
        fig.update_yaxes(range=[0, 1.05])

        roc_thresholds = roc_data.get("thresholds")
        if threshold is not None and isinstance(roc_thresholds, list) and len(roc_thresholds) == len(fpr):
            try:
                diffs = [abs(th - threshold) for th in roc_thresholds]
                best_idx = int(np.argmin(diffs))
                # dashed guides through the chosen point
                fig.add_shape(
                    type="line",
                    x0=fpr[best_idx],
                    x1=fpr[best_idx],
                    y0=0,
                    y1=tpr[best_idx],
                    line=dict(color="gray", width=2, dash="dash"),
                )
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=fpr[best_idx],
                    y0=tpr[best_idx],
                    y1=tpr[best_idx],
                    line=dict(color="gray", width=2, dash="dash"),
                )
                fig.add_trace(
                    go.Scatter(
                        x=[fpr[best_idx]],
                        y=[tpr[best_idx]],
                        mode="markers",
                        marker=dict(color="black", size=10, symbol="x"),
                        name=f"Threshold={threshold}",
                        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{text}<extra></extra>",
                        text=[f"{threshold}"],
                    )
                )
            except Exception as exc:
                print(f"Warning: could not add threshold marker to ROC: {exc}")

        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=f"<span style='font-size:12px;color:#555;'>{pos_label_txt}</span>",
            xanchor="center",
        )

        return _wrap_plot("ROC Curve", fig, config=config)
    except Exception as e:
        print(f"Error building ROC plot: {e}")
        return None


def _build_precision_recall_plot(
    label_stats: dict,
    config: dict,
    threshold: Optional[float] = None,
) -> Optional[Dict[str, str]]:
    """Build Precision-Recall curve directly from test_statistics.json."""
    pr_data = label_stats.get("precision_recall_curve")
    if not isinstance(pr_data, dict):
        return None

    precisions = pr_data.get("precisions")
    recalls = pr_data.get("recalls")
    if not precisions or not recalls or len(precisions) != len(recalls):
        return None

    thresholds = pr_data.get("thresholds")

    try:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines+markers",
                name="Precision-Recall",
                line=dict(color="#d62728", width=4),
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )

        ap_val = (
            label_stats.get("average_precision_macro")
            or label_stats.get("average_precision_micro")
            or label_stats.get("average_precision_samples")
        )
        ap_txt = f" (AP = {ap_val:.3f})" if isinstance(ap_val, (int, float)) else ""

        fig.update_layout(
            title=dict(text=f"Precision-Recall Curve{ap_txt}", x=0.5),
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=700,
            height=600,
            margin=dict(t=80, l=80, r=80, b=80),
            hovermode="closest",
            legend=dict(
                x=0.6,
                y=0.1,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )
        _style_fig(fig)
        fig.update_xaxes(range=[0, 1.0])
        fig.update_yaxes(range=[0, 1.05])

        if threshold is not None and isinstance(thresholds, list) and len(thresholds) == len(recalls):
            try:
                diffs = [abs(th - threshold) for th in thresholds]
                best_idx = int(np.argmin(diffs))
                fig.add_shape(
                    type="line",
                    x0=recalls[best_idx],
                    x1=recalls[best_idx],
                    y0=0,
                    y1=precisions[best_idx],
                    line=dict(color="gray", width=2, dash="dash"),
                )
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=recalls[best_idx],
                    y0=precisions[best_idx],
                    y1=precisions[best_idx],
                    line=dict(color="gray", width=2, dash="dash"),
                )
                fig.add_trace(
                    go.Scatter(
                        x=[recalls[best_idx]],
                        y=[precisions[best_idx]],
                        mode="markers",
                        marker=dict(color="black", size=10, symbol="x"),
                        name=f"Threshold={threshold}",
                        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>Threshold: %{text}<extra></extra>",
                        text=[f"{threshold}"],
                    )
                )
            except Exception as exc:
                print(f"Warning: could not add threshold marker to PR: {exc}")

        return _wrap_plot("Precision-Recall Curve", fig, config=config)
    except Exception as e:
        print(f"Error building Precision-Recall plot: {e}")
        return None


def build_prediction_diagnostics(
    predictions_path: str,
    label_data_path: Optional[str] = None,
    split_value: int = 2,
) -> List[Dict[str, str]]:
    """Generate diagnostic plots from predictions.csv for classification tasks."""
    preds_file = Path(predictions_path)
    if not preds_file.exists():
        return []

    try:
        df_pred = pd.read_csv(predictions_path)
    except Exception as exc:
        print(f"Warning: Unable to read predictions CSV: {exc}")
        return []

    plots: List[Dict[str, str]] = []
    labels_from_dataset: Optional[pd.Series] = None

    filtered_by_split = False

    # If a split column exists, focus on the requested split (e.g., validation=1, test=2).
    # If not, but label_data_path is available and matches row count, use it to filter predictions.
    if SPLIT_COLUMN_NAME in df_pred.columns:
        df_pred = df_pred[df_pred[SPLIT_COLUMN_NAME] == split_value].reset_index(drop=True)
        if df_pred.empty:
            return []
        filtered_by_split = True
    elif label_data_path and Path(label_data_path).exists():
        try:
            df_labels_all = pd.read_csv(label_data_path)

            if SPLIT_COLUMN_NAME in df_labels_all.columns:
                split_mask = (
                    pd.to_numeric(df_labels_all[SPLIT_COLUMN_NAME], errors="coerce")
                    == split_value
                )
                labels_split = df_labels_all.loc[split_mask, LABEL_COLUMN_NAME].reset_index(drop=True)

                if len(df_labels_all) == len(df_pred):
                    df_pred = df_pred.loc[split_mask].reset_index(drop=True)
                    labels_from_dataset = labels_split
                    if df_pred.empty:
                        return []
                    filtered_by_split = True

                elif len(labels_split) == len(df_pred):
                    # predictions.csv is already filtered to this split
                    labels_from_dataset = labels_split
                    filtered_by_split = True

        except Exception as exc:
            print(f"Warning: Unable to filter predictions by split from label data: {exc}")
    # Fallback: no split info available. Assume the predictions file is already filtered
    # (common for test-only exports) and avoid heuristic slicing that could discard rows.
    if not filtered_by_split:
        if split_value != 2:
            return []

    def _maybe_expand_probabilities_column(df: pd.DataFrame, labels_guess: List[str]) -> List[str]:
        """If only a single 'probabilities' column exists (list-like), expand it into per-class columns."""
        if "probabilities" not in df.columns:
            return []
        try:
            # Parse first non-null entry to infer length
            first_val = df["probabilities"].dropna().iloc[0]
            parsed = first_val
            if isinstance(first_val, str):
                parsed = json.loads(first_val)
            probs = list(parsed)
            n = len(probs)
            if n == 0:
                return []
            # Build labels: prefer provided guess; otherwise numeric
            if labels_guess and len(labels_guess) == n:
                labels_use = labels_guess
            else:
                labels_use = [str(i) for i in range(n)]
            # Expand column
            for idx, lbl in enumerate(labels_use):
                df[f"probabilities_{lbl}"] = df["probabilities"].apply(
                    lambda v: (json.loads(v)[idx] if isinstance(v, str) else list(v)[idx]) if pd.notnull(v) else np.nan
                )
            return [f"probabilities_{lbl}" for lbl in labels_use]
        except Exception:
            return []

    # Identify probability columns
    prob_cols = [
        c
        for c in df_pred.columns
        if (
            (c.startswith("label_probabilities_") or c.startswith("probabilities_"))
            and c != "label_probabilities"
        )
    ]
    if not prob_cols and "label_probability" in df_pred.columns:
        prob_cols = ["label_probability"]
    if not prob_cols and "probability" in df_pred.columns:
        prob_cols = ["probability"]
    if not prob_cols and "prediction_probability" in df_pred.columns:
        prob_cols = ["prediction_probability"]
    if not prob_cols and "probabilities" in df_pred.columns:
        labels_guess = (
            sorted([str(u) for u in pd.unique(df_pred[LABEL_COLUMN_NAME])])
            if LABEL_COLUMN_NAME in df_pred.columns
            else []
        )
        prob_cols = _maybe_expand_probabilities_column(df_pred, labels_guess)
    prob_cols_sorted = sorted(prob_cols)

    pos_prob_col, pos_label_hint = _select_positive_probability_column(prob_cols_sorted)
    pos_prob_series = df_pred[pos_prob_col] if pos_prob_col and pos_prob_col in df_pred else None

    # Confidence series: prefer label_probability, otherwise positive prob, otherwise max prob
    confidence_series = None
    if "label_probability" in df_pred.columns:
        confidence_series = df_pred["label_probability"]
    elif pos_prob_series is not None:
        confidence_series = pos_prob_series
    elif prob_cols_sorted:
        confidence_series = df_pred[prob_cols_sorted].max(axis=1)

    # True labels
    def _extract_labels():
        if labels_from_dataset is not None:
            return labels_from_dataset
        candidates = [
            LABEL_COLUMN_NAME,
            f"{LABEL_COLUMN_NAME}_ground_truth",
            f"{LABEL_COLUMN_NAME}__ground_truth",
            f"{LABEL_COLUMN_NAME}_target",
            f"{LABEL_COLUMN_NAME}__target",
            "label",
            "label_true",
        ]
        candidates.extend(
            [
                col
                for col in df_pred.columns
                if (col.startswith(f"{LABEL_COLUMN_NAME}_") or col.startswith(f"{LABEL_COLUMN_NAME}__"))
                and "probabilities" not in col
                and "predictions" not in col
            ]
        )
        for col in candidates:
            if col in df_pred.columns and col not in prob_cols_sorted:
                return df_pred[col]
        if label_data_path and Path(label_data_path).exists():
            try:
                df_all = pd.read_csv(label_data_path)
                if SPLIT_COLUMN_NAME in df_all.columns:
                    df_all = df_all[df_all[SPLIT_COLUMN_NAME] == split_value].reset_index(drop=True)
                if LABEL_COLUMN_NAME in df_all.columns:
                    return df_all[LABEL_COLUMN_NAME].reset_index(drop=True)
            except Exception as exc:
                print(f"Warning: Unable to load labels from dataset: {exc}")
        return None

    labels_series = _extract_labels()

    confidence_plot = None
    if confidence_series is not None:
        fig_conf = go.Figure()
        fig_conf.add_trace(
            go.Histogram(
                x=confidence_series,
                nbinsx=20,
                marker=dict(color="#1f77b4", line=dict(color="#ffffff", width=1)),
                opacity=0.8,
                histnorm="percent",
            )
        )
        fig_conf.update_layout(
            title=dict(text="Prediction Confidence Distribution", x=0.5),
            xaxis_title="Predicted probability (confidence)",
            yaxis_title="Percentage (%)",
            bargap=0.05,
            width=700,
            height=500,
        )
        _style_fig(fig_conf)
        confidence_plot = _wrap_plot("Prediction Confidence Distribution", fig_conf)

    # The remaining plots require true labels and a positive-class probability
    if labels_series is None or pos_prob_series is None:
        if confidence_plot is not None:
            plots.append(confidence_plot)
        return plots

    # Align lengths
    min_len = min(len(labels_series), len(pos_prob_series))
    if min_len == 0:
        if confidence_plot is not None:
            plots.append(confidence_plot)
        return plots
    y_true_raw = labels_series.iloc[:min_len].reset_index(drop=True)
    y_score_series = pd.to_numeric(
        pos_prob_series.iloc[:min_len],
        errors="coerce",
    ).reset_index(drop=True)
    y_score = y_score_series.to_numpy(dtype=float)

    observed_labels = pd.unique(y_true_raw.dropna())
    if len(observed_labels) != 2:
        if confidence_plot is not None:
            plots.append(confidence_plot)
        return plots

    # Determine positive label from the selected probability column suffix.
    positive_label = _resolve_positive_label_from_probability(
        np.array(y_true_raw.dropna()), pos_label_hint
    )
    if positive_label is None:
        if confidence_plot is not None:
            plots.append(confidence_plot)
        return plots

    y_true = (y_true_raw == positive_label).astype(int).values

    # Plot 1: Calibration Curve. Requires exactly two true labels plus valid
    # positive-class probabilities; multiclass and malformed probability exports
    # keep the confidence histogram fallback without adding a reliability curve.
    finite_mask = np.isfinite(y_score) & y_true_raw.notna().to_numpy()
    if finite_mask.any():
        y_score_finite = y_score[finite_mask]
        y_true_finite = y_true[finite_mask]
        if (
            len(np.unique(y_true_finite)) == 2
            and np.all((y_score_finite >= 0.0) & (y_score_finite <= 1.0))
        ):
            try:
                prob_true, prob_pred = calibration_curve(
                    y_true_finite,
                    y_score_finite,
                    n_bins=10,
                    strategy="uniform",
                )
                ece = expected_calibration_error(y_true_finite, y_score_finite, n_bins=10)
                fig_cal = go.Figure()
                fig_cal.add_trace(
                    go.Scatter(
                        x=prob_pred,
                        y=prob_true,
                        mode="lines+markers",
                        name=f"Model - ECE: {ece:.3f}",
                        line=dict(color="#2ca02c", width=4),
                    )
                )
                fig_cal.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect calibration",
                        line=dict(color="gray", width=2, dash="dash"),
                    )
                )
                fig_cal.update_layout(
                    title=dict(text="Calibration Curve", x=0.5),
                    xaxis_title="Mean predicted probability",
                    yaxis_title="Fraction of positives",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    width=700,
                    height=500,
                )
                _style_fig(fig_cal)
                plots.append(
                    _wrap_plot(
                        "Calibration Curve (Test)",
                        fig_cal,
                    )
                )
            except ValueError as exc:
                print(f"Warning: Skipping calibration curve: {exc}")

    # Plot 2: Confidence Histogram
    if confidence_plot is not None:
        plots.append(confidence_plot)

    return plots


def build_binary_threshold_plot(
    predictions_path: str,
    label_data_path: Optional[str] = None,
    split_value: int = 1,
    selected_threshold: Optional[float] = None,
    selected_metric: str = "f1",
) -> Optional[Dict[str, str]]:
    """Build a binary threshold sweep plot (accuracy, precision, recall, F1) for a given split."""
    threshold_data = load_binary_threshold_data(
        predictions_path, label_data_path=label_data_path, split_value=split_value
    )
    if threshold_data is None:
        return None

    y_true_bin = threshold_data["y_true_bin"]
    y_score = threshold_data["y_score"]

    thresholds = np.linspace(0.0, 1.0, 101)
    accs: List[float] = []
    balanced_accs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    mccs: List[float] = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        accs.append(accuracy_score(y_true_bin, preds))
        balanced_accs.append(balanced_accuracy_score(y_true_bin, preds))
        precs.append(precision_score(y_true_bin, preds, zero_division=0))
        recs.append(recall_score(y_true_bin, preds, zero_division=0))
        f1s.append(f1_score(y_true_bin, preds, zero_division=0))
        mccs.append(matthews_corrcoef(y_true_bin, preds))

    if selected_threshold is None:
        optimized = optimize_binary_threshold_values(
            y_true_bin, y_score, metric=selected_metric
        )
        marker_threshold = optimized["threshold"] if optimized else thresholds[int(np.argmax(f1s))]
    else:
        marker_threshold = float(selected_threshold)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=accs, mode="lines", name="Accuracy", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=thresholds, y=balanced_accs, mode="lines", name="Balanced Accuracy", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=thresholds, y=precs, mode="lines", name="Precision", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=thresholds, y=recs, mode="lines", name="Recall", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines", name="F1-Score", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=thresholds, y=mccs, mode="lines", name="MCC", line=dict(width=4)))
    fig.add_shape(
        type="line",
        x0=marker_threshold,
        x1=marker_threshold,
        y0=0,
        y1=1,
        line=dict(color="gray", width=2, dash="dash"),
    )
    fig.update_layout(
        title=dict(text="Threshold plot", x=0.5),
        xaxis_title="Threshold",
        yaxis_title="Metric value",
        yaxis=dict(range=[0, 1]),
        width=760,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _style_fig(fig)
    return _wrap_plot("Threshold plot", fig, include_js=True)


def load_binary_threshold_data(
    predictions_path: str,
    label_data_path: Optional[str] = None,
    split_value: int = 1,
) -> Optional[Dict[str, np.ndarray]]:
    """Load binary labels and positive-class probabilities for threshold sweeps."""
    preds_file = Path(predictions_path)
    if not preds_file.exists():
        return None

    try:
        df_full = pd.read_csv(predictions_path)
    except Exception as exc:
        print(f"Warning: Unable to read predictions CSV for threshold optimization: {exc}")
        return None

    labels_from_dataset: Optional[pd.Series] = None

    if SPLIT_COLUMN_NAME in df_full.columns:
        df_pred = df_full[df_full[SPLIT_COLUMN_NAME] == split_value].reset_index(drop=True)
        if df_pred.empty:
            return None
    else:
        df_pred = df_full.reset_index(drop=True)

    if label_data_path and Path(label_data_path).exists():
        try:
            df_labels_all = pd.read_csv(label_data_path)

            if LABEL_COLUMN_NAME in df_labels_all.columns:
                if SPLIT_COLUMN_NAME in df_labels_all.columns:
                    mask = (
                        pd.to_numeric(df_labels_all[SPLIT_COLUMN_NAME], errors="coerce")
                        == split_value
                    )
                    labels_split = df_labels_all.loc[mask, LABEL_COLUMN_NAME].reset_index(drop=True)

                    if SPLIT_COLUMN_NAME in df_full.columns and len(df_labels_all) == len(df_full):
                        # predictions.csv contains all splits, so filter predictions and labels together
                        df_pred = df_full.loc[mask].reset_index(drop=True)
                        labels_from_dataset = labels_split

                    elif len(labels_split) == len(df_pred):
                        # predictions.csv is already filtered to this split
                        labels_from_dataset = labels_split

                    else:
                        print(
                            "Warning: Unable to align labels for threshold optimization: "
                            f"split labels length={len(labels_split)}, predictions length={len(df_pred)}"
                        )

                elif len(df_labels_all) == len(df_pred):
                    # No split column, but label table and predictions have same row count
                    labels_from_dataset = df_labels_all[LABEL_COLUMN_NAME].reset_index(drop=True)

        except Exception as exc:
            print(f"Warning: Unable to align labels for threshold optimization: {exc}")

    if SPLIT_COLUMN_NAME not in df_full.columns and split_value != 2 and labels_from_dataset is None:
        return None

    prob_cols = [
        c
        for c in df_pred.columns
        if (
            (c.startswith("label_probabilities_") or c.startswith("probabilities_"))
            and c != "label_probabilities"
        )
    ]
    if not prob_cols and "probabilities" in df_pred.columns:
        labels_guess = sorted([str(u) for u in pd.unique(df_pred.get(LABEL_COLUMN_NAME, []))])
        try:
            first_val = df_pred["probabilities"].dropna().iloc[0]
            parsed = json.loads(first_val) if isinstance(first_val, str) else list(first_val)
            n = len(parsed)
            if n > 0:
                labels_use = labels_guess if labels_guess and len(labels_guess) == n else [str(i) for i in range(n)]
                for idx, lbl in enumerate(labels_use):
                    df_pred[f"probabilities_{lbl}"] = df_pred["probabilities"].apply(
                        lambda v: (json.loads(v)[idx] if isinstance(v, str) else list(v)[idx]) if pd.notnull(v) else np.nan
                    )
                prob_cols = [f"probabilities_{lbl}" for lbl in labels_use]
        except Exception:
            prob_cols = []
    prob_cols_sorted = sorted(prob_cols)

    def _strip_prob_prefix(col: str) -> str:
        if col.startswith("label_probabilities_"):
            return col.replace("label_probabilities_", "")
        if col.startswith("probabilities_"):
            return col.replace("probabilities_", "")
        return col

    def _extract_labels():
        if labels_from_dataset is not None:
            return labels_from_dataset
        for col in [
            LABEL_COLUMN_NAME,
            f"{LABEL_COLUMN_NAME}_ground_truth",
            f"{LABEL_COLUMN_NAME}__ground_truth",
            f"{LABEL_COLUMN_NAME}_target",
            f"{LABEL_COLUMN_NAME}__target",
            "label",
            "label_true",
        ]:
            if col in df_pred.columns and col not in prob_cols_sorted:
                return df_pred[col]
        return None

    labels_series = _extract_labels()
    if labels_series is None or not prob_cols_sorted:
        return None

    pos_prob_col, pos_label_hint = _select_positive_probability_column(prob_cols_sorted)
    if pos_prob_col is None:
        return None

    min_len = min(len(labels_series), len(df_pred[pos_prob_col]))
    if min_len == 0:
        return None

    y_true = np.array(labels_series.iloc[:min_len])
    unique_labels = pd.unique(y_true)
    if len(unique_labels) < 2:
        return None
    positive_label = _resolve_positive_label_from_probability(y_true, pos_label_hint)
    if positive_label is None:
        return None
    y_true_bin = (y_true == positive_label).astype(int)
    y_score = np.array(df_pred[pos_prob_col].iloc[:min_len], dtype=float)
    finite_mask = np.isfinite(y_score)
    if not finite_mask.any():
        return None
    finite_labels = y_true[finite_mask]
    negative_labels = [label for label in pd.unique(finite_labels) if label != positive_label]
    negative_label = negative_labels[0] if negative_labels else None

    return {
        "y_true_bin": y_true_bin[finite_mask],
        "y_score": y_score[finite_mask],
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_probability_column": pos_prob_col,
    }


def optimize_binary_threshold_values(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> Optional[Dict[str, float]]:
    """Optimize a binary threshold against a threshold-sensitive metric."""
    metric_key = str(metric or "f1").strip().lower().replace("-", "_").replace(" ", "_")
    display_map = {
        "f1": "F1",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "mcc": "MCC",
        "matthews_correlation_coefficient": "MCC",
    }
    metric_key = "mcc" if metric_key == "matthews_correlation_coefficient" else metric_key
    if metric_key not in display_map:
        raise ValueError(
            "Unsupported threshold optimization metric "
            f"'{metric}'. Valid choices: {', '.join(display_map.values())}."
        )

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(y_score, dtype=float))))
    best_score = None
    best_thresholds: List[float] = []
    for threshold in thresholds:
        preds = (y_score >= threshold).astype(int)
        if metric_key == "f1":
            score = f1_score(y_true_bin, preds, zero_division=0)
        elif metric_key == "accuracy":
            score = accuracy_score(y_true_bin, preds)
        elif metric_key == "balanced_accuracy":
            score = balanced_accuracy_score(y_true_bin, preds)
        elif metric_key == "precision":
            score = precision_score(y_true_bin, preds, zero_division=0)
        elif metric_key == "recall":
            score = recall_score(y_true_bin, preds, zero_division=0)
        elif metric_key == "mcc":
            score = matthews_corrcoef(y_true_bin, preds)
        else:
            continue

        if best_score is None or score > best_score:
            best_score = float(score)
            best_thresholds = [float(threshold)]
        elif score == best_score:
            best_thresholds.append(float(threshold))

    if best_score is None or not best_thresholds:
        return None

    threshold = min(best_thresholds, key=lambda value: abs(value - 0.5))
    return {
        "threshold": float(threshold),
        "metric_value": float(best_score),
        "metric": metric_key,
        "metric_display": display_map[metric_key],
    }


def optimize_binary_threshold_from_predictions(
    predictions_path: str,
    label_data_path: Optional[str] = None,
    split_value: int = 1,
    metric: str = "f1",
) -> Optional[Dict[str, float]]:
    data = load_binary_threshold_data(
        predictions_path, label_data_path=label_data_path, split_value=split_value
    )
    if data is None:
        return None
    return optimize_binary_threshold_values(
        data["y_true_bin"], data["y_score"], metric=metric
    )


def _threshold_metric_display_name(metric: str) -> str:
    metric_key = str(metric or "f1").strip().lower().replace("-", "_").replace(" ", "_")
    display_map = {
        "f1": "F1",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "mcc": "MCC",
    }
    return display_map.get(metric_key, str(metric))


def resolve_binary_threshold_for_report(
    threshold_mode: str,
    requested_metric: str,
    predictions_path: str,
    label_data_path: Optional[str] = None,
    existing_threshold: Optional[float] = None,
    split_value: int = 1,
) -> Dict[str, object]:
    """Resolve the threshold shown in the report for binary classification.

    Prefer recomputing from validation probabilities. If report-time
    predictions are unavailable but Ludwig already persisted an output
    threshold in the experiment description, preserve that threshold instead
    of replacing it with the default 0.5.
    """
    mode = str(threshold_mode or "auto").lower()
    metric_display = _threshold_metric_display_name(requested_metric)

    if mode == "manual":
        threshold = 0.5 if existing_threshold is None else float(existing_threshold)
        return {
            "threshold": threshold,
            "threshold_source": "Manual",
            "threshold_metric": "Not used",
        }

    try:
        optimized_threshold = None
        if predictions_path and Path(predictions_path).exists():
            optimized_threshold = optimize_binary_threshold_from_predictions(
                predictions_path,
                label_data_path=label_data_path,
                split_value=split_value,
                metric=str(requested_metric),
            )
        if optimized_threshold is None:
            raise ValueError("validation probabilities unavailable")
        return {
            "threshold": optimized_threshold["threshold"],
            "threshold_source": "Optimized on validation split",
            "threshold_metric": optimized_threshold["metric_display"],
            "threshold_metric_value": optimized_threshold["metric_value"],
        }
    except Exception as exc:
        if existing_threshold is not None:
            return {
                "threshold": float(existing_threshold),
                "threshold_source": (
                    "Selected during training "
                    "(report validation probabilities unavailable)"
                ),
                "threshold_metric": metric_display,
                "threshold_reason": str(exc),
            }
        return {
            "threshold": 0.5,
            "threshold_source": (
                "Default 0.5 (automatic optimization unavailable)"
            ),
            "threshold_metric": metric_display,
            "threshold_reason": str(exc),
        }


def compute_binary_threshold_metrics_from_predictions(
    predictions_path: str,
    threshold: float,
    label_data_path: Optional[str] = None,
    split_value: int = 2,
) -> Optional[Dict[str, float]]:
    data = load_binary_threshold_data(
        predictions_path, label_data_path=label_data_path, split_value=split_value
    )
    if data is None:
        return None

    y_true_bin = data["y_true_bin"]
    y_pred = (data["y_score"] >= float(threshold)).astype(int)
    tn = int(((y_true_bin == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy_score(y_true_bin, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true_bin, y_pred),
        "precision": precision_score(y_true_bin, y_pred, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true_bin, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true_bin, y_pred),
    }


def build_binary_threshold_classification_plots_from_predictions(
    predictions_path: str,
    threshold: float,
    label_data_path: Optional[str] = None,
    split_value: int = 2,
) -> List[Dict[str, str]]:
    """Build binary test plots from predictions.csv using the selected decision threshold."""
    data = load_binary_threshold_data(
        predictions_path, label_data_path=label_data_path, split_value=split_value
    )
    if data is None:
        return []

    y_true_bin = data["y_true_bin"]
    y_score = data["y_score"]
    y_pred = (y_score >= float(threshold)).astype(int)
    if len(y_true_bin) == 0:
        return []

    positive_label = data.get("positive_label")
    negative_label = data.get("negative_label")
    labels = [
        str(negative_label) if negative_label is not None else f"Not {positive_label}",
        str(positive_label),
    ]
    threshold_text = f"{float(threshold):.3f}"
    common_cfg = {"displayModeBar": True, "scrollZoom": True}
    plots: List[Dict[str, str]] = []

    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
    total = cm.sum()
    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig_cm.update_traces(xgap=2, ygap=2)
    fig_cm.update_layout(
        title=dict(
            text=f"Confusion Matrix (Selected Threshold: {threshold_text})",
            x=0.5,
        ),
        xaxis_title="Predicted",
        yaxis_title="Observed",
        yaxis_autorange="reversed",
        width=600,
        height=600,
        margin=dict(t=100, l=80, r=80, b=80),
    )
    _style_fig(fig_cm)
    mval = cm.max() if cm.size else 0
    color_threshold = mval / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            pct = (value / total * 100) if total > 0 else 0
            color = "white" if value > color_threshold else "black"
            fig_cm.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"<b>{value}</b><br>{pct:.1f}%",
                showarrow=False,
                font=dict(color=color, size=14),
            )
    plots.append(
        _wrap_plot("Confusion Matrix", fig_cm, include_js=True, config=common_cfg)
    )

    if len(np.unique(y_true_bin)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC={roc_auc:.3f})",
                line=dict(color="#1f77b4", width=4),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )
        if len(roc_thresholds) == len(fpr):
            marker_idx = int(np.argmin(np.abs(np.asarray(roc_thresholds) - float(threshold))))
            fig_roc.add_trace(
                go.Scatter(
                    x=[fpr[marker_idx]],
                    y=[tpr[marker_idx]],
                    mode="markers",
                    marker=dict(color="black", size=10, symbol="x"),
                    name=f"Selected threshold={threshold_text}",
                )
            )
        fig_roc.update_layout(
            title=dict(text=f"ROC Curve (Positive Class: {positive_label})", x=0.5),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=700,
            height=600,
            margin=dict(t=80, l=80, r=80, b=80),
        )
        _style_fig(fig_roc)
        fig_roc.update_xaxes(range=[0, 1.0])
        fig_roc.update_yaxes(range=[0, 1.05])
        plots.append(_wrap_plot("ROC Curve", fig_roc, config=common_cfg))

        precision, recall, pr_thresholds = precision_recall_curve(y_true_bin, y_score)
        avg_precision = average_precision_score(y_true_bin, y_score)
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"Precision-Recall (AP={avg_precision:.3f})",
                line=dict(color="#d62728", width=4),
            )
        )
        if len(pr_thresholds) > 0:
            marker_idx = int(np.argmin(np.abs(np.asarray(pr_thresholds) - float(threshold))))
            point_idx = min(marker_idx + 1, len(precision) - 1)
            fig_pr.add_trace(
                go.Scatter(
                    x=[recall[point_idx]],
                    y=[precision[point_idx]],
                    mode="markers",
                    marker=dict(color="black", size=10, symbol="x"),
                    name=f"Selected threshold={threshold_text}",
                )
            )
        fig_pr.update_layout(
            title=dict(text=f"Precision-Recall Curve (Positive Class: {positive_label})", x=0.5),
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=700,
            height=600,
            margin=dict(t=80, l=80, r=80, b=80),
        )
        _style_fig(fig_pr)
        fig_pr.update_xaxes(range=[0, 1.0])
        fig_pr.update_yaxes(range=[0, 1.05])
        plots.append(_wrap_plot("Precision-Recall Curve", fig_pr, config=common_cfg))

    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    def _ratio(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    class_rows = [
        {
            "label": labels[0],
            "precision": _ratio(tn, tn + fn),
            "recall": _ratio(tn, tn + fp),
            "f1_score": _ratio(2 * tn, 2 * tn + fp + fn),
            "accuracy": accuracy_score(y_true_bin, y_pred),
            "matthews_correlation_coefficient": matthews_corrcoef(y_true_bin, y_pred),
            "specificity": _ratio(tp, tp + fn),
        },
        {
            "label": labels[1],
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1_score": _ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy": accuracy_score(y_true_bin, y_pred),
            "matthews_correlation_coefficient": matthews_corrcoef(y_true_bin, y_pred),
            "specificity": _ratio(tn, tn + fp),
        },
    ]
    metrics = [
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "matthews_correlation_coefficient",
        "specificity",
    ]
    z = [[row[m] for m in metrics] for row in class_rows]
    text = [[f"{row[m]:.2f}" for m in metrics] for row in class_rows]
    fig_cr = go.Figure(
        go.Heatmap(
            z=z,
            x=[m.replace("_", " ") for m in metrics],
            y=[row["label"] for row in class_rows],
            text=text,
            texttemplate="%{text}",
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Value"),
        )
    )
    fig_cr.update_layout(
        title=dict(text=f"Per-Class Metrics (Selected Threshold: {threshold_text})", x=0.5),
        xaxis_title="",
        yaxis_title="Class",
        width=700,
        height=520,
        margin=dict(t=80, l=80, r=80, b=80),
    )
    _style_fig(fig_cr)
    plots.append(_wrap_plot("Per-Class metrics", fig_cr, config=common_cfg))

    return plots


def build_multiclass_roc_pr_plots(
    predictions_path: str,
    split_value: int = 2,
) -> List[Dict[str, str]]:
    """Build one-vs-rest ROC and PR curves for multi-class classification from predictions."""
    preds_file = Path(predictions_path)
    if not preds_file.exists():
        return []
    try:
        df_pred = pd.read_csv(predictions_path)
    except Exception as exc:
        print(f"Warning: Unable to read predictions CSV: {exc}")
        return []

    if SPLIT_COLUMN_NAME in df_pred.columns:
        df_pred = df_pred[df_pred[SPLIT_COLUMN_NAME] == split_value].reset_index(drop=True)
    if df_pred.empty:
        return []

    if LABEL_COLUMN_NAME not in df_pred.columns:
        return []

    # Identify per-class probability columns
    prob_cols = [
        c
        for c in df_pred.columns
        if (
            (c.startswith("label_probabilities_") or c.startswith("probabilities_"))
            and c != "label_probabilities"
        )
    ]
    if not prob_cols:
        return []
    labels = [c.replace("label_probabilities_", "").replace("probabilities_", "") for c in prob_cols]
    labels_sorted = sorted(labels)

    # Ensure all labels are present as probability columns
    prob_map = {
        c.replace("label_probabilities_", "").replace("probabilities_", ""): c
        for c in prob_cols
    }
    if len(labels_sorted) < 3:
        return []

    y_true_raw = df_pred[LABEL_COLUMN_NAME].astype(str)
    # Drop rows with NaN probabilities across any class to avoid metric errors
    prob_matrix = df_pred[[prob_map[lbl] for lbl in labels_sorted]].astype(float)
    mask_valid = ~prob_matrix.isnull().any(axis=1)
    prob_matrix = prob_matrix[mask_valid]
    y_true_raw = y_true_raw[mask_valid]
    if prob_matrix.empty:
        return []

    y_true_bin = label_binarize(y_true_raw, classes=labels_sorted)
    y_score = prob_matrix.to_numpy()

    plots: List[Dict[str, str]] = []

    # ROC: one-vs-rest + micro
    fig_roc = go.Figure()
    added_any = False
    for idx, lbl in enumerate(labels_sorted):
        if y_true_bin[:, idx].sum() == 0 or y_true_bin[:, idx].sum() == len(y_true_bin):
            continue  # skip classes without both positives and negatives
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{lbl} (AUC={auc(fpr, tpr):.3f})",
                line=dict(width=3),
            )
        )
        added_any = True
    # Micro-average only if we have mixed labels
    if y_true_bin.sum() > 0 and y_true_bin.sum() < y_true_bin.size:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        fig_roc.add_trace(
            go.Scatter(
                x=fpr_micro,
                y=tpr_micro,
                mode="lines",
                name=f"Micro-average (AUC={auc(fpr_micro, tpr_micro):.3f})",
                line=dict(width=3, dash="dash"),
            )
        )
        added_any = True
    if not added_any:
        return []
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", width=2, dash="dot"),
        )
    )
    fig_roc.update_layout(
        title=dict(text="Multi-class ROC-AUC (one-vs-rest)", x=0.5),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=820,
        height=620,
        legend=dict(
            x=0.62,
            y=0.05,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    _style_fig(fig_roc)
    plots.append(_wrap_plot("Multi-class ROC-AUC (one-vs-rest)", fig_roc))

    # PR: one-vs-rest + micro AP
    fig_pr = go.Figure()
    added_pr = False
    for idx, lbl in enumerate(labels_sorted):
        if y_true_bin[:, idx].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true_bin[:, idx], y_score[:, idx])
        ap = average_precision_score(y_true_bin[:, idx], y_score[:, idx])
        fig_pr.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=f"{lbl} (AP={ap:.3f})",
                line=dict(width=3),
            )
        )
        added_pr = True
    if y_true_bin.sum() > 0:
        prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
        ap_micro = average_precision_score(y_true_bin, y_score, average="micro")
        fig_pr.add_trace(
            go.Scatter(
                x=rec_micro,
                y=prec_micro,
                mode="lines",
                name=f"Micro-average (AP={ap_micro:.3f})",
                line=dict(width=3, dash="dash"),
            )
        )
        added_pr = True
    if not added_pr:
        return plots
    fig_pr.update_layout(
        title=dict(text="Multi-class Precision-Recall (one-vs-rest)", x=0.5),
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=820,
        height=620,
        legend=dict(
            x=0.62,
            y=0.05,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    _style_fig(fig_pr)
    plots.append(_wrap_plot("Multi-class Precision-Recall (one-vs-rest)", fig_pr))

    return plots


def build_multiclass_metric_plots(
    test_stats_path: str,
    metadata_csv_path: Optional[str] = None,
    train_set_metadata_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Alternative multi-class transparency plots using test_statistics.json per-class stats."""
    ts_path = Path(test_stats_path)
    if not ts_path.exists():
        return []
    try:
        with open(ts_path, "r") as f:
            test_stats = json.load(f)
    except Exception:
        return []

    label_stats = test_stats.get("label", {})
    pcs = label_stats.get("per_class_stats", {})
    if not pcs:
        return []
    classes = list(pcs.keys())
    if not classes:
        return []
    friendly_labels = _resolve_confusion_labels(
        label_stats,
        len(classes),
        metadata_csv_path=metadata_csv_path,
        train_set_metadata_path=train_set_metadata_path,
    )
    display_classes = _resolve_display_labels(classes, friendly_labels)

    metrics = ["precision", "recall", "f1_score", "specificity", "accuracy"]
    fig_bar = go.Figure()
    for metric in metrics:
        values = []
        for cls in classes:
            v = pcs.get(cls, {}).get(metric)
            values.append(v if isinstance(v, (int, float)) else 0)
        fig_bar.add_trace(
            go.Bar(
                x=display_classes,
                y=values,
                name=metric.replace("_", " ").title(),
            )
        )
    fig_bar.update_layout(
        title=dict(text="Per-Class Metrics (Test)", x=0.5),
        xaxis_title="Class",
        yaxis_title="Metric value",
        barmode="group",
        width=900,
        height=600,
        legend=dict(
            x=1.02,
            y=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    _style_fig(fig_bar)

    return [_wrap_plot("Per-Class Metrics (Test)", fig_bar)]
