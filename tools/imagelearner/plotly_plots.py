import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative
from constants import LABEL_COLUMN_NAME, SPLIT_COLUMN_NAME
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
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


def build_classification_plots(
    test_stats_path: str,
    training_stats_path: Optional[str] = None,
    metadata_csv_path: Optional[str] = None,
    train_set_metadata_path: Optional[str] = None,
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
    fig_cm.update_layout(
        title=dict(text="Confusion Matrix", x=0.5),
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

    plots.append({
        "title": "Confusion Matrix",
        "html": pio.to_html(
            fig_cm,
            full_html=False,
            include_plotlyjs="cdn",
            config=common_cfg
        )
    })

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
                y=[str(c) for c in classes],
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
        plots.append({
            "title": "Per-Class metrics",
            "html": pio.to_html(
                fig_cr,
                full_html=False,
                include_plotlyjs=False,
                config=common_cfg
            )
        })

    # 3) Prediction Diagnostics (from predictions.csv)
    # Note: appended separately in generate_html_report, not returned here.

    return plots


def build_train_validation_plots(train_stats_path: str) -> List[Dict[str, str]]:
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
        if metric not in stats:
            return []
        vals = stats.get(metric, [])
        if isinstance(vals, list):
            return [float(v) for v in vals]
        try:
            return [float(vals)]
        except Exception:
            return []

    def _line_plot(metric_key: str, title: str, yaxis_title: str) -> Optional[Dict[str, str]]:
        train_series = _get_series(label_train, metric_key)
        val_series = _get_series(label_val, metric_key)
        if not train_series and not val_series:
            return None
        epochs_train = list(range(1, len(train_series) + 1))
        epochs_val = list(range(1, len(val_series) + 1))
        fig = go.Figure()
        if train_series:
            fig.add_trace(
                go.Scatter(
                    x=epochs_train,
                    y=train_series,
                    mode="lines+markers",
                    name="Train",
                    line=dict(width=4),
                )
            )
        if val_series:
            fig.add_trace(
                go.Scatter(
                    x=epochs_val,
                    y=val_series,
                    mode="lines+markers",
                    name="Validation",
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
        return {
            "title": title,
            "html": pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if include_js else False,
            ),
        }

    # Core learning curves
    for key, title in [
        ("roc_auc", "ROC-AUC across epochs"),
        ("precision", "Precision across epochs"),
        ("recall", "Recall/Sensitivity across epochs"),
        ("specificity", "Specificity across epochs"),
    ]:
        plot = _line_plot(key, title, title.replace("Learning Curve", "").strip())
        if plot:
            plots.append(plot)
            include_js = False

    # Precision vs Recall evolution (validation)
    val_prec = _get_series(label_val, "precision")
    val_rec = _get_series(label_val, "recall")
    if val_prec and val_rec:
        epochs = list(range(1, min(len(val_prec), len(val_rec)) + 1))
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=epochs,
                y=val_prec[: len(epochs)],
                mode="lines+markers",
                name="Precision",
            )
        )
        fig_pr.add_trace(
            go.Scatter(
                x=epochs,
                y=val_rec[: len(epochs)],
                mode="lines+markers",
                name="Recall",
            )
        )
        fig_pr.update_layout(
            title=dict(text="Validation Precision and Recall by Epoch", x=0.5),
            xaxis_title="Epoch",
            yaxis_title="Value",
            width=760,
            height=520,
            hovermode="x unified",
        )
        _style_fig(fig_pr)
        plots.append({
            "title": "Precision vs Recall Evolution",
            "html": pio.to_html(
                fig_pr,
                full_html=False,
                include_plotlyjs="cdn" if include_js else False,
            ),
        })
        include_js = False

    # F1-score derived
    def _compute_f1(p: List[float], r: List[float]) -> List[float]:
        f1_vals = []
        for prec, rec in zip(p, r):
            if (prec + rec) == 0:
                f1_vals.append(0.0)
            else:
                f1_vals.append(2 * prec * rec / (prec + rec))
        return f1_vals

    f1_train = _compute_f1(_get_series(label_train, "precision"), _get_series(label_train, "recall"))
    f1_val = _compute_f1(val_prec, val_rec)
    if f1_train or f1_val:
        fig = go.Figure()
        if f1_train:
            fig.add_trace(go.Scatter(x=list(range(1, len(f1_train) + 1)), y=f1_train, mode="lines+markers", name="Train", line=dict(width=4)))
        if f1_val:
            fig.add_trace(go.Scatter(x=list(range(1, len(f1_val) + 1)), y=f1_val, mode="lines+markers", name="Validation", line=dict(width=4)))
        fig.update_layout(
            title=dict(text="F1-Score across epochs (derived)", x=0.5),
            xaxis_title="Epoch",
            yaxis_title="F1-Score",
            width=760,
            height=520,
            hovermode="x unified",
        )
        _style_fig(fig)
        plots.append({
            "title": "F1-Score across epochs (derived)",
            "html": pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if include_js else False,
            ),
        })
        include_js = False

    # Overfitting Gap: Train vs Val ROC-AUC (gap)
    roc_train = _get_series(label_train, "roc_auc")
    roc_val = _get_series(label_val, "roc_auc")
    if roc_train and roc_val:
        epochs_gap = list(range(1, min(len(roc_train), len(roc_val)) + 1))
        gaps = [t - v for t, v in zip(roc_train[:len(epochs_gap)], roc_val[:len(epochs_gap)])]
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(x=epochs_gap, y=gaps, mode="lines+markers", name="Train - Val ROC-AUC", line=dict(width=4)))
        fig_gap.update_layout(
            title=dict(text="Overfitting gap: ROC-AUC across epochs", x=0.5),
            xaxis_title="Epoch",
            yaxis_title="Gap",
            width=760,
            height=520,
            hovermode="x unified",
        )
        _style_fig(fig_gap)
        plots.append({
            "title": "Overfitting gap: ROC-AUC across epochs",
            "html": pio.to_html(
                fig_gap,
                full_html=False,
                include_plotlyjs="cdn" if include_js else False,
            ),
        })
        include_js = False

    # Best Epoch Dashboard (based on max val ROC-AUC)
    if roc_val:
        best_idx = int(np.argmax(roc_val))
        best_epoch = best_idx + 1
        metrics_at_best: Dict[str, Optional[float]] = {}
        # Always include ROC-AUC (used for best-epoch selection)
        metrics_at_best["ROC-AUC"] = roc_val[best_idx] if best_idx < len(roc_val) else None

        # Add commonly available validation metrics when present
        for metric_key, label in [
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("specificity", "Specificity"),
            ("loss", "Loss"),
        ]:
            series = _get_series(label_val, metric_key)
            if series and best_idx < len(series):
                metrics_at_best[label] = series[best_idx]

        # Derived F1 from validation precision/recall if available
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
        plots.append({
            "title": "Best Validation Epoch Snapshot (Metrics)",
            "html": pio.to_html(
                fig_best,
                full_html=False,
                include_plotlyjs="cdn" if include_js else False,
            ),
        })
        include_js = False

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
    epochs_train = list(range(1, len(train_series) + 1))
    epochs_val = list(range(1, len(val_series) + 1))
    fig = go.Figure()
    if train_series:
        fig.add_trace(
            go.Scatter(
                x=epochs_train,
                y=train_series,
                mode="lines+markers",
                name="Train",
                line=dict(width=4),
            )
        )
    if val_series:
        fig.add_trace(
            go.Scatter(
                x=epochs_val,
                y=val_series,
                mode="lines+markers",
                name="Validation",
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
    return {
        "title": title,
        "html": pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if include_js else False,
        ),
    }


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
    epochs = None
    for metric_key, title, ytitle in metrics:
        series = _get_regression_series(label_test, metric_key)
        if not series:
            continue
        if epochs is None:
            epochs = list(range(1, len(series) + 1))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=series[: len(epochs)],
                mode="lines+markers",
                name="Test",
                line=dict(width=4),
            )
        )
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Epoch",
            yaxis_title=ytitle,
            width=760,
            height=520,
            hovermode="x unified",
        )
        _style_fig(fig)
        plots.append({
            "title": title,
            "html": pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if include_js else False,
            ),
        })
        include_js = False
    return plots


def _build_static_roc_plot(
    label_stats: dict, config: dict, friendly_labels: Optional[List[str]] = None
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

        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=f"<span style='font-size:12px;color:#555;'>{pos_label_txt}</span>",
            xanchor="center",
        )

        return {
            "title": "ROC Curve",
            "html": pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                config=config,
            ),
        }
    except Exception as e:
        print(f"Error building ROC plot: {e}")
        return None


def _build_precision_recall_plot(label_stats: dict, config: dict) -> Optional[Dict[str, str]]:
    """Build Precision-Recall curve directly from test_statistics.json."""
    pr_data = label_stats.get("precision_recall_curve")
    if not isinstance(pr_data, dict):
        return None

    precisions = pr_data.get("precisions")
    recalls = pr_data.get("recalls")
    if not precisions or not recalls or len(precisions) != len(recalls):
        return None

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

        return {
            "title": "Precision-Recall Curve",
            "html": pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                config=config,
            ),
        }
    except Exception as e:
        print(f"Error building Precision-Recall plot: {e}")
        return None


def build_prediction_diagnostics(
    predictions_path: str,
    label_data_path: Optional[str] = None,
    split_value: int = 2,
    threshold: Optional[float] = None,
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

    # If a split column exists, focus on the requested split (e.g., validation=1, test=2)
    if SPLIT_COLUMN_NAME in df_pred.columns:
        df_pred = df_pred[df_pred[SPLIT_COLUMN_NAME] == split_value].reset_index(drop=True)
        if df_pred.empty:
            return plots

    plots: List[Dict[str, str]] = []

    def _strip_prob_prefix(col: str) -> str:
        if col.startswith("label_probabilities_"):
            return col.replace("label_probabilities_", "")
        if col.startswith("probabilities_"):
            return col.replace("probabilities_", "")
        return col

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
    if not prob_cols and "probabilities" in df_pred.columns:
        labels_guess = sorted([str(u) for u in pd.unique(df_pred[LABEL_COLUMN_NAME])])
        prob_cols = _maybe_expand_probabilities_column(df_pred, labels_guess)
    prob_cols_sorted = sorted(prob_cols)

    def _select_positive_prob():
        if not prob_cols_sorted:
            return None, None
        # Prefer a column indicating positive/event/true/1
        preferred_keys = ("event", "true", "positive", "pos", "1")
        for col in prob_cols_sorted:
            suffix = _strip_prob_prefix(col).lower()
            if any(k in suffix for k in preferred_keys):
                return col, suffix
        if len(prob_cols_sorted) == 2:
            col = prob_cols_sorted[1]
            return col, _strip_prob_prefix(col)
        col = prob_cols_sorted[0]
        return col, _strip_prob_prefix(col)

    pos_prob_col, pos_label_hint = _select_positive_prob()
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

    # Plot 1: Confidence Histogram
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
        plots.append({
            "title": "Prediction Confidence Distribution",
            "html": pio.to_html(fig_conf, full_html=False, include_plotlyjs=False),
        })

    # The remaining plots require true labels and a positive-class probability
    if labels_series is None or pos_prob_series is None:
        return plots

    # Align lengths
    min_len = min(len(labels_series), len(pos_prob_series))
    if min_len == 0:
        return plots
    y_true_raw = labels_series.iloc[:min_len]
    y_score = np.array(pos_prob_series.iloc[:min_len], dtype=float)

    # Determine positive label
    unique_labels = pd.unique(y_true_raw)
    unique_labels_list = list(unique_labels)
    positive_label = None
    if pos_label_hint and str(pos_label_hint) in [str(u) for u in unique_labels_list]:
        positive_label = pos_label_hint
    elif len(unique_labels_list) == 2:
        positive_label = unique_labels_list[1]
    else:
        positive_label = unique_labels_list[0]

    y_true = (y_true_raw == positive_label).astype(int).values

    # Utility: compute calibration points
    def _calibration_points(y_true_bin: np.ndarray, scores: np.ndarray):
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(scores, bins, right=True)
        bin_centers, frac_positives = [], []
        for b in range(1, len(bins)):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            bin_centers.append(scores[mask].mean())
            frac_positives.append(y_true_bin[mask].mean())
        return bin_centers, frac_positives

    # Plot 2: Calibration Curve (multi-class aware; one-vs-rest per label)
    label_prob_map = {}
    for col in prob_cols_sorted:
        if col.startswith("label_probabilities_"):
            cls = col.replace("label_probabilities_", "")
            label_prob_map[cls] = col

    unique_label_strs = [str(u) for u in unique_labels_list]
    if len(label_prob_map) > 1 and len(unique_label_strs) > 2:
        # Skip multi-class calibration curve for now (not informative in current report)
        pass
    else:
        # Binary/unknown fallback (previous behavior)
        bin_centers, frac_positives = _calibration_points(y_true, y_score)
        if bin_centers and frac_positives:
            fig_cal = go.Figure()
            fig_cal.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=frac_positives,
                    mode="lines+markers",
                    name="Calibration",
                    line=dict(color="#2ca02c", width=4),
                )
            )
            fig_cal.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(color="gray", width=2, dash="dash"),
                )
            )
            fig_cal.update_layout(
                title=dict(text="Calibration Curve", x=0.5),
                xaxis_title="Predicted probability",
                yaxis_title="Observed frequency",
                width=700,
                height=500,
            )
            _style_fig(fig_cal)
            plots.append({
                "title": "Calibration Curve (Predicted Probability vs Observed Frequency)",
                "html": pio.to_html(fig_cal, full_html=False, include_plotlyjs=False),
            })

    # Utility: metric curves vs thresholds
    def _threshold_curves(y_true_bin: np.ndarray, scores: np.ndarray):
        if scores.size == 0:
            return [], [], [], [], []
        s_min, s_max = float(np.nanmin(scores)), float(np.nanmax(scores))
        if not np.isfinite(s_min) or not np.isfinite(s_max):
            return [], [], [], [], []
        if s_min == s_max:
            thresholds = np.linspace(0.0, 1.0, 21)
        else:
            thresholds = np.linspace(max(0.0, s_min), min(1.0, s_max), 21)
        thresholds = thresholds.tolist()
        accs, f1s, sens, specs = [], [], [], []
        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            tp = np.sum((y_true_bin == 1) & (y_pred == 1))
            tn = np.sum((y_true_bin == 0) & (y_pred == 0))
            fp = np.sum((y_true_bin == 0) & (y_pred == 1))
            fn = np.sum((y_true_bin == 1) & (y_pred == 0))
            acc = (tp + tn) / max(len(y_true_bin), 1)
            prec = tp / max(tp + fp, 1e-9)
            rec = tp / max(tp + fn, 1e-9)
            f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            sensitivity = rec
            specificity = tn / max(tn + fp, 1e-9)
            accs.append(acc)
            f1s.append(f1)
            sens.append(sensitivity)
            specs.append(specificity)
        return thresholds, accs, f1s, sens, specs

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
    plots.append({
        "title": "Multi-class ROC-AUC (one-vs-rest)",
        "html": pio.to_html(fig_roc, full_html=False, include_plotlyjs=False),
    })

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
    plots.append({
        "title": "Multi-class Precision-Recall (one-vs-rest)",
        "html": pio.to_html(fig_pr, full_html=False, include_plotlyjs=False),
    })

    return plots


def build_multiclass_metric_plots(test_stats_path: str) -> List[Dict[str, str]]:
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

    metrics = ["precision", "recall", "f1_score", "specificity", "accuracy"]
    fig_bar = go.Figure()
    for metric in metrics:
        values = []
        for cls in classes:
            v = pcs.get(cls, {}).get(metric)
            values.append(v if isinstance(v, (int, float)) else 0)
        fig_bar.add_trace(
            go.Bar(
                x=classes,
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

    return [
        {
            "title": "Per-Class Metrics (Test)",
            "html": pio.to_html(fig_bar, full_html=False, include_plotlyjs=False),
        },
    ]
