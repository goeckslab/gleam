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
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import learning_curve as skl_learning_curve
from sklearn.preprocessing import label_binarize
from utils import (
    get_html_template,
    get_html_closing,
    build_tabbed_html,
)

# Matplotlib / SHAP only where interactivity is limited or APIs are tight
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# =========================
# Utilities
# =========================

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
    y_true: Sequence,
    y_pred: Sequence,
    classes: Optional[Sequence] = None,
    title: str = "Confusion Matrix",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Interactive confusion matrix heatmap (Plotly).
    """
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    ztext = cm.astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=ztext,
            texttemplate="%{text}",
            hovertemplate="Pred: %{x}<br>True: %{y}<br>Count: %{z}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        yaxis_autorange="reversed",
    )
    _save_plotly(fig, path)
    return fig


def generate_roc_curve_plot(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Binary ROC curve (Plotly). y_true_bin must be 0/1.
    """
    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
    auc = roc_auc_score(y_true_bin, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance")
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.65, y=0.1),
    )
    _save_plotly(fig, path)
    return fig


def generate_pr_curve_plot(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision–Recall Curve",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Binary PR curve (Plotly). y_true_bin must be 0/1.
    """
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
    ap = average_precision_score(y_true_bin, y_prob)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode="lines", name=f"AP = {ap:.3f}")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, 1]),
        legend=dict(x=0.65, y=0.1),
        template="plotly_white",
    )
    _save_plotly(fig, path)
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
        title=title,
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
    n_points: int = 201,
) -> go.Figure:
    """
    Precision, recall, F1, and queue rate vs discrimination threshold.
    Draws a dashed vertical line at the F1-optimal threshold and labels it.
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    th = np.linspace(0, 1, n_points)

    precisions, recalls, f1s, qrates = [], [], [], []
    for t in th:
        yhat = (y_prob >= t).astype(int)
        tp = int((yhat & (y_true_bin == 1)).sum())
        fp = int((yhat & (y_true_bin == 0)).sum())
        fn = int(((1 - yhat) & (y_true_bin == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        qrat = float(yhat.mean())

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        qrates.append(qrat)

    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    f1s        = np.array(f1s)
    qrates     = np.array(qrates)

    # F1-optimal threshold
    best_idx = int(np.argmax(f1s))
    t_star = float(th[best_idx])

    fig = go.Figure()
    def add_curve(y, name):
        # line + soft fill to mimic the reference style
        fig.add_trace(go.Scatter(
            x=th, y=y, name=name, mode="lines",
            line=dict(width=3),
            fill="tozeroy", opacity=0.15, hovertemplate="t=%{x:.2f}<br>%{y:.3f}<extra></extra>"
        ))
    add_curve(precisions, "precision")
    add_curve(recalls, "recall")
    add_curve(f1s, "F₁")
    add_curve(qrates, "queue rate")

    # Vertical dashed line at t* (F1-optimal)
    fig.add_vline(x=t_star, line_width=2, line_dash="dash", line_color="black")
    fig.add_annotation(
        x=t_star, y=0.98, xref="x", yref="paper",
        showarrow=False, text=f"t* = {t_star:.2f}",
        font=dict(color="black")
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(title="discrimination threshold", range=[0, 1], gridcolor="#eee"),
        yaxis=dict(title="score", range=[0, 1], gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=50, r=20, t=60, b=50),
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
        title=title,
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
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cls} (AUC {auc:.2f})"))

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False)
    )
    fig.update_layout(
        title=title,
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
        title=title,
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
    fig = px.bar(df_m, x="Metric", y="Score", color="Phase", barmode="group", title=title)
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

    fig = px.scatter(x=y_true, y=y_pred, opacity=0.6, labels={"x": "Actual", "y": "Predicted"}, title=title)
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
                     title=title)
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
    fig = px.histogram(x=residuals, nbins=bins, labels={"x": "Residual"}, title=title)
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
        title=title,
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
        title=title,
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
                       title=title)
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
        title=title,
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
    perf_table_html: str | None = None,   # <— NEW
) -> str:
    y_true = df_train[label_column].values

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

    pieces: list[str] = []

    # 0) Model Performance Summary (no Test) — FIRST
    if perf_table_html:
        pieces.append(f"<div class='card'>{perf_table_html}</div>")

    # 1) Learning Curve — Accuracy
    if problem_type in ("binary", "multiclass") and pred_labels is not None:
        fig_acc = generate_learning_curve_from_predictions(
            y_true=y_true, y_pred=np.asarray(pred_labels),
            metric="accuracy", title="Learning Curves Label Accuracy", seed=seed
        )
        pieces.append(fig_acc.to_html(full_html=False, include_plotlyjs="cdn"))

    # 2) Learning Curve — Loss
    if problem_type in ("binary", "multiclass") and pred_proba is not None:
        classes = np.unique(y_true)
        pp = pred_proba.reshape(-1) if pred_proba.ndim == 1 or (pred_proba.ndim == 2 and pred_proba.shape[1] == 1) else pred_proba
        fig_ll = generate_learning_curve_from_predictions(
            y_true=y_true, y_proba=pp, classes=classes,
            metric="log_loss", title="Learning Curves Label Loss", seed=seed
        )
        pieces.append(fig_ll.to_html(full_html=False, include_plotlyjs=False))

    # 3) Threshold Plot (binary only)
    if problem_type == "binary" and pred_proba is not None:
        pos_scores = pred_proba.reshape(-1) if pred_proba.ndim == 1 else pred_proba[:, -1]
        y_bin = (y_true == np.max(np.unique(y_true))).astype(int)
        fig_thr = generate_threshold_plot(
            y_true_bin=y_bin, y_prob=pos_scores, title="Threshold Plot"
        )
        pieces.append(fig_thr.to_html(full_html=False, include_plotlyjs=False))

    if not pieces:
        return "<h2>Training Diagnostics</h2><p><em>No training diagnostics available for this run.</em></p>"

    return "<h2>Training Diagnostics</h2>" + "".join(f"<div class='plotly-center'>{p}</div>" for p in pieces)

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
        title=title,
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
    Compatible with both TabularPredictor (supports `silent`) and MultiModalPredictor (no `silent`).
    """
    def _evaluate(df):
        try:
            return predictor.evaluate(df, silent=True)  # TabularPredictor path
        except TypeError:
            return predictor.evaluate(df)               # MultiModalPredictor path

    train_scores = _safe_floatify(_evaluate(df_train))
    val_scores   = _safe_floatify(_evaluate(df_val))
    test_scores  = _safe_floatify(_evaluate(df_test))
    return train_scores, val_scores, test_scores

def build_summary_html(
    predictor,
    args,
    problem_type: str,
    train_scores: Dict[str, float],
    val_scores: Dict[str, float],
    test_scores: Dict[str, float],
    tmpdir: str,
) -> str:
    """
    A compact summary: metrics table (train/val/test) + a tiny run/config block.
    Returns HTML (to be used in the 'Validation Summary & Config' tab).
    """
    # metrics table
    metrics = sorted(set(train_scores) | set(val_scores) | set(test_scores))
    head = "<thead><tr><th>Metric</th><th>Train</th><th>Validation</th><th>Test</th></tr></thead>"
    rows = []
    for m in metrics:
        tr = train_scores.get(m, np.nan)
        vr = val_scores.get(m, np.nan)
        te = test_scores.get(m, np.nan)
        def fmt(x): 
            try: return f"{float(x):.4f}"
            except: return ""
        rows.append(f"<tr><td>{m.replace('_',' ').title()}</td><td>{fmt(tr)}</td><td>{fmt(vr)}</td><td>{fmt(te)}</td></tr>")
    metrics_tbl = f"<h2>Model Performance Summary</h2><table class='performance-summary'>{head}<tbody>{''.join(rows)}</tbody></table>"

    # simple run/config info
    cfg_lines = [
        ("Problem type", problem_type),
        ("Target column", args.label_column),
        ("Image column", args.image_column or "—"),
        ("Time limit (s)", args.time_limit if getattr(args, 'time_limit', None) else "—"),
        ("Random seed", args.random_seed),
    ]
    cfg_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in cfg_lines)
    cfg_tbl = f"<h2>Run Configuration</h2><table>{cfg_rows}</table>"

    return metrics_tbl + "<hr>" + cfg_tbl


def build_test_html_and_plots(
    predictor,
    problem_type: str,
    df_test: pd.DataFrame,
    label_column: str,
    tmpdir: str,
) -> Tuple[str, List[str]]:
    """
    Create a test-summary section (with a placeholder for metric rows) and a list of Plotly HTML divs.
    Returns: (html_template_with_{}, list_of_plot_divs)
    """
    plots: List[str] = []

    y_true = df_test[label_column].values
    # Try proba/labels where meaningful
    pred_labels = None
    pred_proba  = None
    try:
        pred_labels = predictor.predict(df_test)
    except Exception:
        pass
    try:
        # TabularPredictor/MultiModalPredictor both expose predict_proba for classification
        pred_proba = predictor.predict_proba(df_test)
    except Exception:
        pred_proba = None

    # Classification visuals
    if problem_type in ("binary", "multiclass") and pred_labels is not None:
        # Confusion matrix
        fig_cm = generate_confusion_matrix_plot(y_true, pred_labels, title="Confusion Matrix")
        plots.append(fig_cm.to_html(full_html=False, include_plotlyjs="cdn"))

        # Per-class metrics (bar)
        fig_pc = generate_per_class_metrics_plot(y_true, pred_labels, title="Per-Class Metrics")
        plots.append(fig_pc.to_html(full_html=False, include_plotlyjs=False))

        # ROC/PR where possible
        if pred_proba is not None:
            # Normalize outputs to (n_samples, n_classes)
            if isinstance(pred_proba, pd.Series):
                proba_arr = pred_proba.to_numpy().reshape(-1, 1)
            elif isinstance(pred_proba, pd.DataFrame):
                proba_arr = pred_proba.to_numpy()
            else:
                proba_arr = np.asarray(pred_proba)

            classes = np.unique(y_true)
            if problem_type == "binary":
                # accept shape (n,) or (n,2)
                if proba_arr.ndim == 1 or proba_arr.shape[1] == 1:
                    y_bin = (y_true == classes.max()).astype(int)
                    fig_roc = generate_roc_curve_plot(y_bin, proba_arr.reshape(-1), title="ROC Curve")
                    plots.append(fig_roc.to_html(full_html=False, include_plotlyjs=False))

                    fig_pr = generate_pr_curve_plot(y_bin, proba_arr.reshape(-1), title="Precision–Recall Curve")
                    plots.append(fig_pr.to_html(full_html=False, include_plotlyjs=False))
                else:
                    # take positive class (assume index of max class)
                    pos_idx = 1 if proba_arr.shape[1] > 1 else 0
                    y_bin = (y_true == classes.max()).astype(int)
                    fig_roc = generate_roc_curve_plot(y_bin, proba_arr[:, pos_idx], title="ROC Curve")
                    plots.append(fig_roc.to_html(full_html=False, include_plotlyjs=False))

                    fig_pr = generate_pr_curve_plot(y_bin, proba_arr[:, pos_idx], title="Precision–Recall Curve")
                    plots.append(fig_pr.to_html(full_html=False, include_plotlyjs=False))
            else:
                # multiclass one-vs-rest curves
                fig_mroc = generate_multiclass_roc_curve_plot(y_true, proba_arr, classes=classes, title="Multiclass ROC Curves")
                plots.append(fig_mroc.to_html(full_html=False, include_plotlyjs=False))

                fig_mpr = generate_multiclass_pr_curve_plot(y_true, proba_arr, classes=classes, title="Multiclass PR Curves")
                plots.append(fig_mpr.to_html(full_html=False, include_plotlyjs=False))

    # Regression visuals
    if problem_type == "regression":
        if pred_labels is None:
            pred_labels = predictor.predict(df_test)
        fig_sc = generate_scatter_plot(y_true, pred_labels, title="Predicted vs Actual")
        plots.append(fig_sc.to_html(full_html=False, include_plotlyjs="cdn"))

        fig_res = generate_residual_plot(y_true, pred_labels, title="Residual Plot")
        plots.append(fig_res.to_html(full_html=False, include_plotlyjs=False))

        fig_hist = generate_residual_histogram(y_true, pred_labels, title="Residual Histogram")
        plots.append(fig_hist.to_html(full_html=False, include_plotlyjs=False))

        fig_cal = generate_regression_calibration_plot(y_true, pred_labels, title="Regression Calibration")
        plots.append(fig_cal.to_html(full_html=False, include_plotlyjs=False))

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
    df_test: pd.DataFrame,
    label_column: str,
    tmpdir: str,
    random_seed: int,
) -> str:
    """
    Feature importance for Tabular; for Multimodal we show a placeholder if not supported.
    Returns HTML (Feature Importance tab).
    """
    try:
        # TabularPredictor supports feature_importance
        imp = predictor.feature_importance(df_test)
        # Expect columns: 'feature', 'importance'
        if "feature" in imp.columns and "importance" in imp.columns:
            top = imp.head(30)
            fig = px.bar(top, x="feature", y="importance", title="Top Feature Importances")
            fig.update_layout(xaxis_tickangle=45, template="plotly_white")
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
        else:
            # AutoGluon older versions return a Series
            s = imp if isinstance(imp, pd.Series) else pd.Series(imp)
            top = s.sort_values(ascending=False).head(30)
            fig = px.bar(top.reset_index(), x="index", y=0, title="Top Feature Importances")
            fig.update_layout(xaxis_title="feature", yaxis_title="importance",
                              xaxis_tickangle=45, template="plotly_white")
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception:
        # MultimodalPredictor or unsupported
        return "<p><em>Feature importance not available for this predictor.</em></p>"


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
    # Append plots under the Test tab
    test_full = test_html + "".join(f"<div class='plotly-center'>{p}</div>" for p in plots)

    tabs = build_tabbed_html(summary_html, train_html, test_full, feature_html, explainer_html=None)

    html = get_html_template()
    html += """
<style>
  .plotly-center { display: flex; justify-content: center; }
  .plotly-center .plotly-graph-div, .plotly-center .js-plotly-plot { margin: 0 auto !important; }
  .js-plotly-plot, .plotly-graph-div { margin-left: auto !important; margin-right: auto !important; }
</style>
"""
    html += tabs
    html += get_html_closing()
    return html
