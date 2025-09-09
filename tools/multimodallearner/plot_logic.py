from __future__ import annotations

import os
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
)
from sklearn.model_selection import learning_curve as skl_learning_curve
from sklearn.preprocessing import label_binarize

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
    title: str = "Threshold Curve",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Binary threshold sweep plotting Precision/Recall/F1 vs threshold (Plotly).
    """
    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_prob)
    thresholds = np.append(thresholds, 1.0)  # Align lengths
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision, mode="lines", name="Precision"))
    fig.add_trace(go.Scatter(x=thresholds, y=recall, mode="lines", name="Recall"))
    fig.add_trace(go.Scatter(x=thresholds, y=f1, mode="lines", name="F1"))
    fig.update_layout(
        title=title,
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    _save_plotly(fig, path)
    return fig


def generate_per_class_metrics_plot(
    y_true: Sequence,
    y_pred: Sequence,
    metrics: Sequence[str] = ("precision", "recall", "f1-score"),
    title: str = "Per-Class Metrics",
    path: Optional[str] = None,
) -> go.Figure:
    """
    Per-class metrics bar chart (Plotly), using sklearn classification_report.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [
        c for c in report.keys()
        if c not in {"accuracy", "macro avg", "micro avg", "weighted avg"}
    ]
    df = (
        pd.DataFrame(report).T.loc[classes, list(metrics)]
        .reset_index()
        .rename(columns={"index": "Class"})
    )
    df_m = df.melt(id_vars="Class", var_name="Metric", value_name="Score")
    fig = px.bar(df_m, x="Class", y="Score", color="Metric", barmode="group", title=title)
    fig.update_yaxes(range=[0, 1])
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
