from __future__ import annotations

import os
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import learning_curve as skl_learning_curve
from sklearn.preprocessing import label_binarize
from report_utils import (
    get_html_template,
    get_html_closing,
    build_tabbed_html,
)

# Matplotlib / SHAP only where interactivity is limited or APIs are tight
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from html import escape as _escape
import html

# =========================
# Utilities
# =========================
def plot_with_table_style_title(fig, title: str) -> str:
    """
    Render a Plotly figure with a report-style <h2> header so it matches the
    green table section headers.
    """
    # kill Plotly’s built-in title
    fig.update_layout(title=None)

    # figure HTML without PlotlyJS (we load it once globally)
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False)

    # use <h2> — your CSS already styles <h2> like the table headers
    return f"""
<h2>{html.escape(title)}</h2>
<div class="plotly-center">{plot_html}</div>
""".strip()

def _save_plotly(fig: go.Figure, path: Optional[str]) -> None:
    """
    Save a Plotly figure. If `path` ends with `.html`, save interactive HTML.
    If it ends with a raster extension (png/jpg/jpeg/webp), uses Kaleido.
    If None, do nothing (caller may choose to display in notebook).
    """
    if not path:
        return
    ext = os.path.splitext(path)[1].lower()
    if ext == ".html":
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    else:
        # Requires kaleido: pip install -U kaleido
        fig.write_image(path)


def _save_matplotlib(path: Optional[str]) -> None:
    """Save current Matplotlib figure if `path` is provided, else show()."""
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =========================
# Classification Plots
# =========================

def generate_confusion_matrix_plot(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
) -> go.Figure:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Class order (works for strings or numbers)
    labels = pd.Index(np.unique(np.concatenate([y_true, y_pred])), dtype=object).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Use categorical axes by passing string labels for x/y
    cats = [str(l) for l in labels]
    total = int(cm.sum())
    
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=cats,              # categorical x
            y=cats,              # categorical y
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
            xgap=2,
            ygap=2,
            hovertemplate="True=%{y}<br>Pred=%{x}<br>Count=%{z}<extra></extra>",
            zmin=0
        )
    )

    # Add annotations with count and percentage (all white text, matching sample_output.html)
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            pct = (val / total * 100) if total > 0 else 0
            # All text is white (matching sample_output.html)
            text_color = "white"
            # Count annotation (bold, bottom)
            annotations.append(
                dict(
                    x=cats[j],
                    y=cats[i],
                    text=f"<b>{val}</b>",
                    showarrow=False,
                    font=dict(color=text_color, size=14),
                    xanchor="center",
                    yanchor="bottom",
                    yshift=2
                )
            )
            # Percentage annotation (top)
            annotations.append(
                dict(
                    x=cats[j],
                    y=cats[i],
                    text=f"{pct:.1f}%",
                    showarrow=False,
                    font=dict(color=text_color, size=13),
                    xanchor="center",
                    yanchor="top",
                    yshift=-2
                )
            )

    fig.update_layout(
        title=None,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        xaxis=dict(type="category"),
        yaxis=dict(type="category", autorange="reversed"),  # typical CM orientation
        margin=dict(l=80, r=20, t=40, b=80),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=annotations
    )
    return fig

def generate_roc_curve_plot(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    marker_threshold: float | None = None,
) -> go.Figure:
    y_true_bin = np.asarray(y_true_bin).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    fpr, tpr, thr = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))

    # 45° chance line (no legend to keep it clean)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash"), showlegend=False))

    # Optional marker at the user threshold
    if marker_threshold is not None and len(thr):
        # roc_curve returns thresholds of same length as fpr/tpr; includes inf at idx 0
        finite = np.isfinite(thr)
        if np.any(finite):
            idx_local = int(np.argmin(np.abs(thr[finite] - float(marker_threshold))))
            idx = int(np.nonzero(finite)[0][idx_local])  # map back to original indices
            x_m, y_m = float(fpr[idx]), float(tpr[idx])

            fig.add_trace(
                go.Scatter(
                    x=[x_m], y=[y_m],
                    mode="markers",
                    name=f"@ {float(marker_threshold):.2f}",
                    marker=dict(size=10)
                )
            )

    fig.update_layout(
        title=None,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend=dict(x=1, y=0, xanchor="right"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def generate_pr_curve_plot(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision–Recall Curve",
    marker_threshold: float | None = None,
) -> go.Figure:
    y_true_bin = np.asarray(y_true_bin).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    precision, recall, thr = precision_recall_curve(y_true_bin, y_score)
    pr_auc = auc(recall, precision)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))

    # Optional marker at the user threshold
    if marker_threshold is not None and len(thr):
        # In PR, thresholds has length len(precision)-1. The point for thr[j] is (recall[j+1], precision[j+1]).
        j = int(np.argmin(np.abs(thr - float(marker_threshold))))
        j = int(np.clip(j, 0, len(thr) - 1))
        x_m, y_m = float(recall[j + 1]), float(precision[j + 1])

        fig.add_trace(
            go.Scatter(
                x=[x_m], y=[y_m],
                mode="markers",
                name=f"@ {float(marker_threshold):.2f}",
                marker=dict(size=10)
            )
        )

    fig.update_layout(
        title=None,
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend=dict(x=1, y=0, xanchor="right"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig

def generate_calibration_plot(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Binary calibration curve (Plotly).
    """
    prob_true, prob_pred = calibration_curve(y_true_bin, y_prob, n_bins=n_bins, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Perfect")
    )
    fig.update_layout(
        title=None,
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Probability",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


def generate_threshold_plot(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Threshold Plot",
    user_threshold: float | None = None,
) -> go.Figure:
    y_true = np.asarray(y_true_bin, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()

    # Use uniform threshold grid for smoother curves (0 to 1 in steps of 0.01)
    th = np.linspace(0.0, 1.0, 101)

    prec, rec, f1, qrate = [], [], [], []
    for t in th:
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y_true == 1)).sum())
        fp = int(((yhat == 1) & (y_true == 0)).sum())
        fn = int(((yhat == 0) & (y_true == 1)).sum())

        pr = tp / (tp + fp) if (tp + fp) else np.nan  # undefined when no predicted positives
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f  = (2 * pr * rc) / (pr + rc) if (pr + rc) and not np.isnan(pr) else 0.0
        q  = float(yhat.mean())

        prec.append(pr)
        rec.append(rc)
        f1.append(f)
        qrate.append(q)

    # Choose t* where F1 is maximized (ignore NaN precision rows)
    f1_arr = np.asarray(f1, dtype=float)
    # Filter out NaN values for finding optimal threshold
    valid_f1_mask = ~np.isnan(f1_arr)
    if np.any(valid_f1_mask):
        best_idx = int(np.nanargmax(f1_arr))
        t_star = float(th[best_idx])
    else:
        t_star = 0.5  # fallback

    # Replace NaNs for plotting (set to 0 where precision is undefined)
    prec_plot = np.nan_to_num(prec, nan=0.0)

    fig = go.Figure()
    
    # Precision (blue line)
    fig.add_trace(go.Scatter(
        x=th, y=prec_plot, mode="lines", name="Precision",
        line=dict(width=3, color="#1f77b4"),
        hovertemplate="Threshold=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>"
    ))
    
    # Recall (orange line)
    fig.add_trace(go.Scatter(
        x=th, y=rec, mode="lines", name="Recall",
        line=dict(width=3, color="#ff7f0e"),
        hovertemplate="Threshold=%{x:.3f}<br>Recall=%{y:.3f}<extra></extra>"
    ))
    
    # F1 (green line)
    fig.add_trace(go.Scatter(
        x=th, y=f1_arr, mode="lines", name="F1",
        line=dict(width=3, color="#2ca02c"),
        hovertemplate="Threshold=%{x:.3f}<br>F1=%{y:.3f}<extra></extra>"
    ))
    
    # Queue Rate (grey dashed line)
    fig.add_trace(go.Scatter(
        x=th, y=qrate, mode="lines", name="Queue Rate",
        line=dict(width=2, color="#808080", dash="dash"),
        hovertemplate="Threshold=%{x:.3f}<br>Queue Rate=%{y:.3f}<extra></extra>"
    ))

    # F1*-optimal threshold marker (dashed vertical line)
    fig.add_vline(
        x=t_star, 
        line_width=2, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"t* = {t_star:.2f}",
        annotation_position="top"
    )

    # User threshold (solid red line) if provided
    if user_threshold is not None:
        fig.add_vline(
            x=float(user_threshold), 
            line_width=2, 
            line_color="red",
            annotation_text=f"threshold = {float(user_threshold):.2f}",
            annotation_position="top"
        )

    fig.update_layout(
        title=None,
        template="plotly_white",
        xaxis=dict(
            title="Discrimination Threshold", 
            range=[0, 1], 
            gridcolor="#e0e0e0",
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Score", 
            range=[0, 1], 
            gridcolor="#e0e0e0",
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1.0
        ),
        margin=dict(l=60, r=20, t=40, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

def generate_per_class_metrics_plot(
    y_true: Sequence,
    y_pred: Sequence,
    metrics: Sequence[str] = ("precision", "recall", "f1_score"),
    title: str = "Classification Report",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Per-class metrics heatmap (Plotly), similar to sklearn's classification report.
    Rows = classes, columns = metrics; cell text shows the value (0–1).
    """
    # Map display names -> sklearn keys
    key_map = {"f1_score": "f1-score", "precision": "precision", "recall": "recall"}
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    # Order classes sensibly (numeric if possible, else lexical)
    def _sort_key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))

    # Use all classes seen in y_true or y_pred (so rows don't jump around)
    uniq = sorted(set(list(y_true) + list(y_pred)), key=_sort_key)
    classes = [str(c) for c in uniq]

    # Build Z matrix (rows=classes, cols=metrics)
    used_metrics = [key_map.get(m, m) for m in metrics]
    z = []
    for c in classes:
        row = report.get(c, {})
        z.append([float(row.get(m, 0.0) or 0.0) for m in used_metrics])
    z = np.array(z, dtype=float)

    # Pretty cell labels
    z_text = [[f"{v:.2f}" for v in r] for r in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(metrics),          # keep display names ("precision", "recall", "f1_score")
            y=classes,                # classes as strings
            colorscale="Reds",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Value"),
            text=z_text,
            texttemplate="%{text}",
            hovertemplate="Class %{y}<br>%{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=None,
        xaxis_title="",
        yaxis_title="Class",
        template="plotly_white",
        margin=dict(l=60, r=60, t=70, b=40),
    )

    _save_plotly(fig, path)
    return fig


def generate_multiclass_roc_curve_plot(
    y_true: Sequence,
    y_prob: np.ndarray,
    classes: Sequence,
    title: str = "Multiclass ROC Curve",
    path: Optional[str] = None,
) -> go.Figure:
    """
    One-vs-rest ROC curves for multiclass (Plotly).
    Handles binary passed as 2-column probs as well.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Normalize to shape (n_samples, n_classes)
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        y_prob = np.hstack([1 - y_prob.reshape(-1, 1), y_prob.reshape(-1, 1)])

    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1 and y_prob.shape[1] == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    if y_prob.shape[1] != y_true_bin.shape[1]:
        raise ValueError(
            f"Shape mismatch: y_prob has {y_prob.shape[1]} columns but y_true_bin has {y_true_bin.shape[1]}."
        )

    fig = go.Figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cls} (AUC {auc_val:.2f})"))

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False)
    )
    fig.update_layout(
        title=None,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


def generate_multiclass_pr_curve_plot(
    y_true: Sequence,
    y_prob: np.ndarray,
    classes: Optional[Sequence] = None,
    title: str = "Precision–Recall Curve",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Multiclass PR curves (Plotly). If classes is None or len==2, shows binary PR.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig = go.Figure()

    if not classes or len(classes) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"AP = {ap:.2f}"))
    else:
        for i, cls in enumerate(classes):
            y_true_bin = (y_true == cls).astype(int)
            y_prob_cls = y_prob[:, i]
            precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_cls)
            ap = average_precision_score(y_true_bin, y_prob_cls)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{cls} (AP {ap:.2f})"))

    fig.update_layout(
        title=None,
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


def generate_metric_comparison_bar(
    metrics_scores: Mapping[str, Sequence[float]],
    phases: Sequence[str] = ("train", "val", "test"),
    title: str = "Metric Comparison Across Phases",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Grouped bar chart comparing metrics across phases (Plotly).
    metrics_scores: {metric_name: [train, val, test]}
    """
    df = pd.DataFrame(metrics_scores, index=phases).T.reset_index().rename(columns={"index": "Metric"})
    df_m = df.melt(id_vars="Metric", var_name="Phase", value_name="Score")
    fig = px.bar(df_m, x="Metric", y="Score", color="Phase", barmode="group", title=None)
    ymax = max(1.0, df_m["Score"].max() * 1.05)
    fig.update_yaxes(range=[0, ymax])
    fig.update_layout(template="plotly_white")
    _save_plotly(fig, path)
    return fig


# =========================
# Regression Plots
# =========================

def generate_scatter_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Predicted vs Actual",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Predicted vs. Actual scatter with y=x reference (Plotly).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vmin = float(min(np.min(y_true), np.min(y_pred)))
    vmax = float(max(np.max(y_true), np.max(y_pred)))

    fig = px.scatter(x=y_true, y=y_pred, opacity=0.6, labels={"x": "Actual", "y": "Predicted"}, title=None)
    fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", line=dict(dash="dash"), name="Ideal"))
    fig.update_layout(template="plotly_white")
    _save_plotly(fig, path)
    return fig


def generate_residual_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Residual Plot",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Residuals vs Predicted (Plotly).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig = px.scatter(x=y_pred, y=residuals, opacity=0.6,
                     labels={"x": "Predicted", "y": "Residual (Actual - Predicted)"},
                     title=None)
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white")
    _save_plotly(fig, path)
    return fig


def generate_residual_histogram(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    bins: int = 30,
    title: str = "Residual Histogram",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Residuals histogram (Plotly).
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig = px.histogram(x=residuals, nbins=bins, labels={"x": "Residual"}, title=None)
    fig.update_layout(yaxis_title="Frequency", template="plotly_white")
    _save_plotly(fig, path)
    return fig


def generate_regression_calibration_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    num_bins: int = 10,
    title: str = "Regression Calibration Plot",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Binned Actual vs Predicted means (Plotly).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    bins = np.array_split(np.arange(len(y_pred_sorted)), num_bins)
    bin_means_pred = [float(np.mean(y_pred_sorted[idx])) for idx in bins if len(idx)]
    bin_means_true = [float(np.mean(y_true_sorted[idx])) for idx in bins if len(idx)]

    vmin = float(min(np.min(y_pred), np.min(y_true)))
    vmax = float(max(np.max(y_pred), np.max(y_true)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bin_means_pred, y=bin_means_true, mode="lines+markers",
                             name="Binned Actual vs Predicted"))
    fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", line=dict(dash="dash"),
                             name="Ideal"))
    fig.update_layout(
        title=None,
        xaxis_title="Mean Predicted per bin",
        yaxis_title="Mean Actual per bin",
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


# =========================
# Confidence / Diagnostics
# =========================

def plot_error_vs_confidence(
    y_true: Union[Sequence[int], np.ndarray],
    y_proba: Union[Sequence[float], np.ndarray],
    n_bins: int = 10,
    title: str = "Error vs Confidence",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Error rate vs confidence (binary), confidence=max(p, 1-p). Plotly.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba).reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)
    confidence = np.maximum(y_proba, 1 - y_proba)
    error = (y_pred != y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(confidence, bins, right=True)

    centers, err_rates = [], []
    for i in range(1, len(bins)):
        mask = (idx == i)
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            err_rates.append(float(error[mask].mean()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers, y=err_rates, mode="lines+markers", name="Error rate"))
    fig.update_layout(
        title=None,
        xaxis_title="Confidence (max predicted probability)",
        yaxis_title="Error Rate",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


def plot_confidence_histogram(
    y_proba: np.ndarray,
    bins: int = 20,
    title: str = "Confidence Histogram",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Histogram of max predicted probabilities (Plotly).
    Works for binary (n_samples,) or (n_samples,2) and multiclass (n_samples,C).
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        confidences = np.maximum(y_proba, 1 - y_proba)
    else:
        confidences = np.max(y_proba, axis=1)

    fig = px.histogram(x=confidences, nbins=bins, range_x=(0, 1),
                       labels={"x": "Confidence (max predicted probability)"},
                       title=None)
    fig.update_layout(yaxis_title="Count", template="plotly_white")
    _save_plotly(fig, path)
    return fig


# =========================
# Learning Curve
# =========================


def generate_learning_curve_from_predictions(
    y_true,
    y_pred=None,
    y_proba=None,
    classes=None,
    metric: str = "accuracy",
    train_fracs: np.ndarray = np.linspace(0.1, 1.0, 10),
    n_repeats: int = 5,
    seed: int = 42,
    title: str = "Learning Curve",
    path: str | None = None,
) -> go.Figure:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    N = len(y_true)

    if metric == "accuracy" and y_pred is None:
        raise ValueError("accuracy curve requires y_pred")
    if metric == "log_loss" and y_proba is None:
        raise ValueError("log_loss curve requires y_proba")

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
    if y_pred is not None:
        y_pred = np.asarray(y_pred)

    sizes = (np.clip((train_fracs * N).astype(int), 1, N)).tolist()
    means, stds = [], []
    for n in sizes:
        vals = []
        for _ in range(n_repeats):
            idx = rng.choice(N, size=n, replace=False)
            if metric == "accuracy":
                vals.append(float((y_true[idx] == y_pred[idx]).mean()))
            else:
                if y_proba.ndim == 1:
                    p = y_proba[idx]
                    pp = np.column_stack([1 - p, p])
                else:
                    pp = y_proba[idx]
                vals.append(float(log_loss(y_true[idx], pp, labels=None if classes is None else classes)))
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=means, mode="lines+markers", name=("training" if metric == "log_loss" else "learning"),
        line=dict(width=3, shape="spline"), marker=dict(size=7),
        error_y=dict(type="data", array=stds, visible=True)
    ))
    fig.update_layout(
        title=None,
        template="plotly_white",
        xaxis=dict(title="epoch" if metric == "log_loss" else "samples", gridcolor="#eee"),
        yaxis=dict(title=("loss" if metric == "log_loss" else "accuracy"), gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    if path:
        _save_plotly(fig, path)
    return fig

def build_train_html_and_plots(
    predictor,
    problem_type: str,
    df_train: pd.DataFrame,
    label_column: str,
    tmpdir: str,
    seed: int = 42,
    perf_table_html: str | None = None,
    threshold: Optional[float] = None,
    section_tile: str = "Training Diagnostics",
) -> str:
    y_true = df_train[label_column].values
    threshold = None
    # predictions on TRAIN
    pred_labels, pred_proba = None, None
    try:
        pred_labels = predictor.predict(df_train)
    except Exception:
        pass
    try:
        proba_raw = predictor.predict_proba(df_train)
        pred_proba = proba_raw.to_numpy() if isinstance(proba_raw, (pd.Series, pd.DataFrame)) else np.asarray(proba_raw)
    except Exception:
        pred_proba = None

    if problem_type == "binary" and threshold is not None and pred_proba is not None:
        classes = np.unique(y_true)
        pos_label, neg_label = classes.max(), classes.min()
        pos_scores = pred_proba.reshape(-1) if pred_proba.ndim == 1 or pred_proba.shape[1] == 1 else pred_proba[:, -1]
        pred_labels = np.where(pos_scores >= float(threshold), pos_label, neg_label)

    pieces: list[str] = []

    # 0) Model Performance Summary (no Test) — FIRST
    if perf_table_html:
        pieces.append(f"<div class='card'>{perf_table_html}</div>")

    # 1) Learning Curve — Accuracy
    if problem_type in ("binary", "multiclass") and pred_labels is not None:
        fig_acc = generate_learning_curve_from_predictions(
            y_true=y_true, y_pred=np.asarray(pred_labels),
            metric="accuracy", title="Learning Curves — Label Accuracy", seed=seed
        )
        pieces.append(plot_with_table_style_title(fig_acc, "Learning Curves — Label Accuracy"))

    # 2) Learning Curve — Loss
    if problem_type in ("binary", "multiclass") and pred_proba is not None:
        classes = np.unique(y_true)
        pp = pred_proba.reshape(-1) if pred_proba.ndim == 1 or (pred_proba.ndim == 2 and pred_proba.shape[1] == 1) else pred_proba
        fig_ll = generate_learning_curve_from_predictions(
            y_true=y_true, y_proba=pp, classes=classes,
            metric="log_loss", title="Learning Curves — Label Loss", seed=seed
        )
        pieces.append(plot_with_table_style_title(fig_ll, "Learning Curves — Label Loss"))

    # 3) Threshold Plot (binary only)
    if problem_type == "binary" and pred_proba is not None:
        pos_scores = pred_proba.reshape(-1) if pred_proba.ndim == 1 else pred_proba[:, -1]
        y_bin = (y_true == np.max(np.unique(y_true))).astype(int)
        fig_thr = generate_threshold_plot(y_true_bin=y_bin, y_prob=pos_scores, title="Threshold Plot",
                                          user_threshold=threshold)
        pieces.append(plot_with_table_style_title(fig_thr, "Threshold Plot"))

    if not pieces:
        return "<h2>Training Diagnostics</h2><p><em>No training diagnostics available for this run.</em></p>"

    return "<h2>Train/Validation Performance Summary</h2>" + "".join(pieces)

def generate_learning_curve(
    estimator,
    X,
    y,
    scoring: str = "r2",
    cv_folds: int = 5,
    n_jobs: int = -1,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    title: str = "Learning Curve",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Learning curve using sklearn.learning_curve, visualized with Plotly.
    """
    sizes, train_scores, test_scores = skl_learning_curve(
        estimator, X, y, cv=cv_folds, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=train_mean, mode="lines+markers", name="Training score",
        error_y=dict(type="data", array=train_std, visible=True)
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=test_mean, mode="lines+markers", name="CV score",
        error_y=dict(type="data", array=test_std, visible=True)
    ))
    fig.update_layout(
        title=None,
        xaxis_title="Training examples",
        yaxis_title=scoring,
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


# =========================
# SHAP (Matplotlib-based)
# =========================

def generate_shap_summary_plot(
    shap_values, features: pd.DataFrame, title: str = "SHAP Summary Plot", path: Optional[str] = None
) -> None:
    """
    SHAP summary plot (Matplotlib). SHAP's interactive support with Plotly is limited;
    keep matplotlib for clarity and stability.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.title(title)
    _save_matplotlib(path)


def generate_shap_force_plot(
    explainer, instance: pd.DataFrame, title: str = "SHAP Force Plot", path: Optional[str] = None
) -> None:
    """
    SHAP force plot (Matplotlib).
    """
    shap_values = explainer(instance)
    plt.figure(figsize=(10, 4))
    shap.plots.force(shap_values[0], show=False)
    plt.title(title)
    _save_matplotlib(path)


def generate_shap_waterfall_plot(
    explainer, instance: pd.DataFrame, title: str = "SHAP Waterfall Plot", path: Optional[str] = None
) -> None:
    """
    SHAP waterfall plot (Matplotlib).
    """
    shap_values = explainer(instance)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(title)
    _save_matplotlib(path)


def infer_problem_type(predictor, df_train_full: pd.DataFrame, label_column: str) -> str:
    """
    Return 'binary', 'multiclass', or 'regression'.
    Prefer the predictor's own metadata when available; otherwise infer from label dtype/uniques.
    """
    # AutoGluon predictors usually expose .problem_type; be defensive.
    pt = getattr(predictor, "problem_type", None)
    if isinstance(pt, str):
        pt_l = pt.lower()
        if "regression" in pt_l:
            return "regression"
        if "binary" in pt_l:
            return "binary"
        if "multiclass" in pt_l or "multiclass" in pt_l:
            return "multiclass"

    y = df_train_full[label_column]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return "regression"
    return "binary" if y.nunique() == 2 else "multiclass"


def _safe_floatify(d: Dict[str, Any]) -> Dict[str, float]:
    """Make evaluate() outputs JSON/csv friendly floats."""
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            # keep only real-valued scalars
            pass
    return out


def evaluate_all(
    predictor,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_column: str,
    problem_type: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Run predictor.evaluate on train/val/test and normalize the result dicts to floats.
    MultiModalPredictor does not accept the `silent` kwarg, so call defensively.
    """
    def _evaluate(df):
        try:
            return predictor.evaluate(df, silent=True)
        except TypeError:
            return predictor.evaluate(df)

    train_scores = _safe_floatify(_evaluate(df_train))
    val_scores   = _safe_floatify(_evaluate(df_val))
    test_scores  = _safe_floatify(_evaluate(df_test))
    return train_scores, val_scores, test_scores

def build_summary_html(
    predictor,
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame],
    df_test: Optional[pd.DataFrame],
    label_column: str,
    extra_run_rows: Optional[list[tuple[str, str]]] = None,
    class_balance_html: Optional[str] = None,
    perf_table_html: Optional[str] = None,
    feature_html: Optional[str] = None,
) -> str:
    sections = []
    
    # Performance Summary
    if perf_table_html:
        sections.append(f"""
<section class="section">
  <h2 class="section-title">Model Performance Summary</h2>
  <div class="card">
    {perf_table_html}
  </div>
</section>
""".strip())

    # Model Configuration

    # Remove Predictor type and Framework, and ensure Model Architecture is present
    base_rows: list[tuple[str, str]] = []
    if extra_run_rows:
        # Remove any rows with keys 'Predictor type' or 'Framework'
        base_rows.extend([(k, v) for (k, v) in extra_run_rows if k not in ("Predictor type", "Framework")])

    def _fmt(v):
        if v is None or v == "":
            return "—"
        return _escape(str(v))

    rows_html = "\n".join(
        f"<tr><td>{_escape(str(k))}</td><td>{_fmt(v)}</td></tr>"
        for k, v in base_rows
    )

    sections.append(f"""
<section class="section">
  <h2 class="section-title">Model Configuration</h2>
  <div class="card">
    <table class="kv-table">
      <thead><tr><th>Key</th><th>Value</th></tr></thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</section>
""".strip())

    # Dataset Overview
    if class_balance_html:
        sections.append(f"""
<section class="section">
  <h2 class="section-title">Dataset Overview</h2>
  <div class="card">
    {class_balance_html}
  </div>
</section>
""".strip())

    # Feature Importance
    if feature_html:
        sections.append(f"""
<section class="section">
  <h2 class="section-title">Feature Importance Analysis</h2>
  <div class="card">
    {feature_html}
  </div>
</section>
""".strip())

    return "\n".join(sections).strip()

def build_feature_importance_html(predictor, df_train: pd.DataFrame, label_column: str) -> str:
    """Build a visualization of feature importance."""
    try:
        # Try to get feature importance from predictor
        fi = None
        if hasattr(predictor, "feature_importance") and callable(predictor.feature_importance):
            try:
                fi = predictor.feature_importance(df_train)
            except Exception as e:
                return f"<p>Could not compute feature importance: {e}</p>"

        if fi is None or (isinstance(fi, pd.DataFrame) and fi.empty):
            return "<p>Feature importance not available for this model.</p>"

        # Format as a sortable table
        rows = []
        if isinstance(fi, pd.DataFrame):
            fi = fi.sort_values("importance", ascending=False)
            for _, row in fi.iterrows():
                feat = row.index[0] if isinstance(row.index, pd.Index) else row["feature"]
                imp = float(row["importance"])
                rows.append(f"<tr><td>{_escape(str(feat))}</td><td>{imp:.4f}</td></tr>")
        else:
            # Handle other formats (dict, etc)
            for feat, imp in sorted(fi.items(), key=lambda x: float(x[1]), reverse=True):
                rows.append(f"<tr><td>{_escape(str(feat))}</td><td>{float(imp):.4f}</td></tr>")

        if not rows:
            return "<p>No feature importance values available.</p>"

        table_html = f"""
        <table class="performance-summary">
          <thead>
            <tr>
              <th class="sortable">Feature</th>
              <th class="sortable">Importance</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
        """
        return table_html
        
    except Exception as e:
        return f"<p>Error building feature importance visualization: {e}</p>"

def build_test_html_and_plots(
    predictor,
    problem_type: str,
    df_test: pd.DataFrame,
    label_column: str,
    tmpdir: str,
    threshold: Optional[float] = None,
) -> Tuple[str, List[str]]:
    """
    Create a test-summary section (with a placeholder for metric rows) and a list of Plotly HTML divs.
    Returns: (html_template_with_{}, list_of_plot_divs)
    """
    plots: List[str] = []

    y_true = df_test[label_column].values
    classes = np.unique(y_true)

    # Try proba/labels where meaningful
    pred_labels = None
    pred_proba  = None
    try:
        pred_labels = predictor.predict(df_test)
    except Exception:
        pass
    try:
        # MultiModalPredictor exposes predict_proba for classification problems.
        pred_proba = predictor.predict_proba(df_test)
    except Exception:
        pred_proba = None

    proba_arr = None
    if pred_proba is not None:
        if isinstance(pred_proba, pd.Series):
            proba_arr = pred_proba.to_numpy().reshape(-1, 1)
        elif isinstance(pred_proba, pd.DataFrame):
            proba_arr = pred_proba.to_numpy()
        else:
            proba_arr = np.asarray(pred_proba)

    # Thresholded labels for binary
    if problem_type == "binary" and threshold is not None and proba_arr is not None:
        pos_label, neg_label = classes.max(), classes.min()
        pos_scores = proba_arr.reshape(-1) if (proba_arr.ndim == 1 or proba_arr.shape[1] == 1) else proba_arr[:, -1]
        pred_labels = np.where(pos_scores >= float(threshold), pos_label, neg_label)

    # Confusion matrix / per-class now reflect thresholded labels
    if problem_type in ("binary", "multiclass") and pred_labels is not None:
        fig_cm = generate_confusion_matrix_plot(y_true, pred_labels, title="Confusion Matrix")
        plots.append(plot_with_table_style_title(fig_cm, "Confusion Matrix"))

        fig_pc = generate_per_class_metrics_plot(y_true, pred_labels, title="Per-Class Metrics")
        plots.append(plot_with_table_style_title(fig_pc, "Per-Class Metrics"))

        # ROC/PR where possible — choose positive-class scores safely
        pos_label = classes.max()  # or set explicitly, e.g., 1 or "yes"

        if isinstance(pred_proba, pd.DataFrame):
            proba_arr = pred_proba.to_numpy()
            if pos_label in pred_proba.columns:
                pos_idx = list(pred_proba.columns).index(pos_label)
            else:
                pos_idx = -1  # fallback to last column
        elif isinstance(pred_proba, pd.Series):
            proba_arr = pred_proba.to_numpy().reshape(-1, 1)
            pos_idx = 0
        else:
            proba_arr = np.asarray(pred_proba) if pred_proba is not None else None
            pos_idx = -1 if (proba_arr is not None and proba_arr.ndim == 2 and proba_arr.shape[1] > 1) else 0

        if proba_arr is not None:
            y_bin = (y_true == pos_label).astype(int)
            pos_scores = (
                proba_arr.reshape(-1)
                if proba_arr.ndim == 1 or proba_arr.shape[1] == 1
                else proba_arr[:, pos_idx]
            )

            fig_roc = generate_roc_curve_plot(y_bin, pos_scores, title="ROC Curve", marker_threshold=threshold)
            plots.append(plot_with_table_style_title(fig_roc, f"ROC Curve{'' if threshold is None else f' (marker at threshold={threshold:.2f})'}"))

            fig_pr = generate_pr_curve_plot(y_bin, pos_scores, title="Precision–Recall Curve", marker_threshold=threshold)
            plots.append(plot_with_table_style_title(fig_pr, f"Precision–Recall Curve{'' if threshold is None else f' (marker at threshold={threshold:.2f})'}"))

    # Regression visuals
    if problem_type == "regression":
        if pred_labels is None:
            pred_labels = predictor.predict(df_test)
        fig_sc = generate_scatter_plot(y_true, pred_labels, title="Predicted vs Actual")
        plots.append(plot_with_table_style_title(fig_sc, "Predicted vs Actual"))

        fig_res = generate_residual_plot(y_true, pred_labels, title="Residual Plot")
        plots.append(plot_with_table_style_title(fig_res, "Residual Plot"))

        fig_hist = generate_residual_histogram(y_true, pred_labels, title="Residual Histogram")
        plots.append(plot_with_table_style_title(fig_hist, "Residual Histogram"))

        fig_cal = generate_regression_calibration_plot(y_true, pred_labels, title="Regression Calibration")
        plots.append(plot_with_table_style_title(fig_cal, "Regression Calibration"))

    # Small HTML template with placeholder for metric rows the caller fills in
    test_html_template = """
      <h2>Test Performance Summary</h2>
      <table class="performance-summary">
        <thead><tr><th>Metric</th><th>Test</th></tr></thead>
        <tbody>{}</tbody>
      </table>
    """
    return test_html_template, plots


def build_feature_html(
    predictor,
    df_train: pd.DataFrame,
    label_column: str,
    include_modalities: bool = True,       # ← NEW
    include_class_balance: bool = True,    # ← NEW
) -> str:
    sections = []

    # (Typical feature importance content…)
    fi_html = build_feature_importance_html(predictor, df_train, label_column)
    sections.append(f"<section class='section'><h2 class='section-title'>Feature Importance</h2><div class='card'>{fi_html}</div></section>")

    # Previously: Modalities & Inputs and/or Class Balance may have been here.
    # Only render them if flags are True.
    if include_modalities:
        from report_utils import build_modalities_html
        modalities_html = build_modalities_html(predictor, df_train, label_column)
        sections.append(f"<section class='section'><h2 class='section-title'>Modalities & Inputs</h2><div class='card'>{modalities_html}</div></section>")

    if include_class_balance:
        from report_utils import build_class_balance_html
        cb_html = build_class_balance_html(df_train, label_column)
        sections.append(f"<section class='section'><h2 class='section-title'>Class Balance (Train Full)</h2><div class='card'>{cb_html}</div></section>")

    return "\n".join(sections)


def assemble_full_html_report(
    summary_html: str,
    train_html: str,
    test_html: str,
    plots: List[str],
    feature_html: str,
) -> str:
    """
    Wrap the four tabs using utils.build_tabbed_html and return full HTML.
    """
    # Append plots under the Test tab (already wrapped with titles)
    test_full = test_html + "".join(plots)

    tabs = build_tabbed_html(summary_html, train_html, test_full, feature_html, explainer_html=None)

    html_out = get_html_template()

    # 🔧 Ensure Plotly JS is available (we render plots with include_plotlyjs=False)
    html_out += '\n<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>\n'

    # Optional: centering tweaks
    html_out += """
<style>
  .plotly-center { display: flex; justify-content: center; }
  .plotly-center .plotly-graph-div, .plotly-center .js-plotly-plot { margin: 0 auto !important; }
  .js-plotly-plot, .plotly-graph-div { margin-left: auto !important; margin-right: auto !important; }
</style>
"""
    html_out += tabs
    html_out += get_html_closing()
    return html_out
