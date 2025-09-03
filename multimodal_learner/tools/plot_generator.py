import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

#save plotly figure to the given path
def save_plot(fig, path):
    if path:
        fig.write_image(path)

def generate_confusion_matrix_plot(
    y_true, y_pred, classes=None, title="Confusion Matrix", path=None
):
    """Generate and save a confusion matrix heatmap using Plotly."""
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=cm.astype(str),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )
    save_plot(fig, path)


def generate_roc_curve_plot(
    y_true_bin, y_prob, title="ROC Curve", path=None
):
    """Generate and save an ROC curve plot using Plotly. Assumes y_true_bin is binary 0/1."""
    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
    roc_auc = roc_auc_score(y_true_bin, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    save_plot(fig, path)

def generate_pr_curve_plot(
    y_true_bin, y_prob, title="Precision-Recall Curve", path=None
):
    """Generate and save a Precision-Recall curve plot using Plotly. Assumes y_true_bin is binary 0/1."""
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
    ap = average_precision_score(y_true_bin, y_prob)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            line_shape="hv",
            name=f"AP = {ap:.2f}",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    save_plot(fig, path)


def generate_calibration_plot(
    y_true_bin, y_prob, n_bins=10, title="Calibration Plot", path=None
):
    """Generate and save a calibration plot using Plotly. Assumes y_true_bin is binary 0/1."""
    prob_true, prob_pred = calibration_curve(
        y_true_bin, y_prob, n_bins=n_bins, strategy="uniform"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Perfect",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Probability",
    )
    save_plot(fig, path)


def generate_per_class_metrics_plot(
    y_true,
    y_pred,
    metrics=["precision", "recall", "f1-score"],
    title="Per-Class Metrics",
    path=None,
):
    """Generate and save a bar plot of per-class metrics using Plotly."""
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [
        cls
        for cls in report
        if cls not in ["accuracy", "macro avg", "micro avg", "weighted avg"]
    ]
    df = pd.DataFrame(report).T.loc[classes, metrics].reset_index().rename(
        columns={"index": "Class"}
    )
    df_long = df.melt(id_vars="Class", var_name="Metric", value_name="Score")
    fig = px.bar(
        df_long,
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        title=title,
    )
    fig.update_yaxes(range=[0, 1])
    save_plot(fig, path)


def generate_scatter_plot(
    y_true, y_pred, title="Predicted vs Actual", path=None
):
    """Generate and save a scatter plot of predicted vs actual values using Plotly."""
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        opacity=0.5,
        labels={"x": "Actual", "y": "Predicted"},
        title=title,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash"),
            name="Perfect",
        )
    )
    save_plot(fig, path)


def generate_residual_plot(
    y_true, y_pred, title="Residual Plot", path=None
):
    """Generate and save a residual plot using Plotly."""
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        opacity=0.5,
        labels={"x": "Predicted", "y": "Residual (Actual - Predicted)"},
        title=title,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    save_plot(fig, path)


def generate_residual_histogram(
    y_true, y_pred, bins=30, title="Residual Histogram", path=None
):
    """Generate and save a histogram of residuals using Plotly."""
    residuals = y_true - y_pred
    fig = px.histogram(
        x=residuals,
        nbins=bins,
        labels={"x": "Residual"},
        title=title,
    )
    fig.update_layout(yaxis_title="Frequency")
    save_plot(fig, path)


def generate_metric_comparison_bar(
    metrics_scores,
    phases=["train", "val", "test"],
    title="Metric Comparison Across Phases",
    path=None,
):
    """Generate and save a bar plot comparing metrics across phases using Plotly."""
    df = pd.DataFrame(metrics_scores, index=phases).T.reset_index().rename(
        columns={"index": "Metric"}
    )
    df_long = df.melt(id_vars="Metric", var_name="Phase", value_name="Score")
    fig = px.bar(
        df_long,
        x="Metric",
        y="Score",
        color="Phase",
        barmode="group",
        title=title,
    )
    max_score = df_long["Score"].max()
    fig.update_yaxes(range=[0, max(max_score, 1) * 1.1])
    save_plot(fig, path)

def generate_shap_summary_plot(
    shap_values, features, title="SHAP Summary Plot", path=None
):
    """Generate and save a SHAP summary plot."""
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

def generate_shap_force_plot(
    explainer, instance, title="SHAP Force Plot", path=None
):
    """Generate and save a SHAP force plot for a single instance."""
    shap_values = explainer(instance)
    fig = plt.figure(figsize=(10, 4))
    shap.plots.force(shap_values[0], show=False)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def generate_shap_waterfall_plot(
    explainer, instance, title="SHAP Waterfall Plot", path=None
):
    """Generate and save a SHAP waterfall plot for a single instance."""
    shap_values = explainer(instance)
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

def generate_threshold_plot(
    y_true_bin, y_prob, title="Threshold Curve", path=None
):
    """Generate and save a threshold plot showing precision, recall, and F1-score
    """
    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_prob)
    thresholds = np.append(thresholds, 1.0)  # Add 1.0 for plotting
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid div by 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision, mode="lines", name="Precision"))
    fig.add_trace(go.Scatter(x=thresholds, y=recall, mode="lines", name="Recall"))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode="lines", name="F1 Score"))

    fig.update_layout(
        title=title,
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.7, y=0.3),
    )
    if path:
        fig.write_image(path)
    else:
        fig.show()

import numpy as np

def plot_error_vs_confidence(y_true, y_proba, n_bins=10, title="Error vs Confidence", save_path=None):

    """Plots Error vs. Confidence"""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    confidence = np.maximum(y_proba, 1 - y_proba)
    error = (y_pred != y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidence, bins, right=True)

    bin_centers = []
    bin_error_rates = []

    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_centers.append(np.mean(confidence[bin_mask]))
            bin_error_rates.append(np.mean(error[bin_mask]))

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, bin_error_rates, marker='o', linestyle='-')
    plt.xlabel("Confidence (max predicted probability)")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_confidence_histogram(y_proba, bins=20, title="Confidence Histogram", save_path=None):
    
    """Plots the maximum predicted probabilities (confidence)."""

    if y_proba.ndim == 1:

        # Binary classification
        confidences = np.maximum(y_proba, 1 - y_proba)
    else:
        # Multiclass
        confidences = np.max(y_proba, axis=1)

    plt.figure(figsize=(8, 6))
    plt.hist(confidences, bins=bins, range=(0, 1), edgecolor="black", alpha=0.75)
    plt.xlabel("Confidence (max predicted probability)")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def generate_learning_curve(
    obj, feat_matrix, targ_vec, scoring="r2", cv_folds=5, num_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
    title="Learning Curve", path=None
):
    """
    Generate and save a learning curve for a regression model.
    
    Parameters:
  
    obj : The regressor object
    feat_matrix : Feature matrix.
    targ_vec : Target vector.
    scoring : Scoring metric to evaluate (default is 'r2').
    cv_folds : Number of cross-validation folds.
    num_jobs : Number of jobs running in parallel
    train_sizes : numbers of training examples
    title : Plot title
    path : plot saved here
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(
        obj, feat_matrix, targ_vec, cv=cv_folds, scoring=scoring, n_jobs=num_jobs, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.grid(True)

    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")

    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def generate_multiclass_roc_curve_plot(y_true, y_prob, classes, title="Multiclass ROC Curve", path=None):
   
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        y_prob = np.hstack([1 - y_prob.reshape(-1, 1), y_prob.reshape(-1, 1)])

    
    y_true_bin = label_binarize(y_true, classes=classes)

    if y_true_bin.shape[1] == 1 and y_prob.shape[1] == 2:
    
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    fig = go.Figure()

    
    if y_prob.shape[1] != y_true_bin.shape[1]:
        raise ValueError(f"Shape mismatch: y_prob has {y_prob.shape[1]} columns but y_true_bin has {y_true_bin.shape[1]} columns.")

    
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Class {class_label} (AUC = {roc_auc:.2f})"
        ))

    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    if path:
        fig.write_image(path)
    else:
        fig.show()

    
def generate_multiclass_pr_curve_plot(
    y_true, y_prob, classes=None, title="Precision-Recall Curve", path=None
):
        """Generate and save a Precision-Recall curve plot"""

        fig = go.Figure()

        if classes is None or len(classes) == 2:
    
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            ap = average_precision_score(y_true, y_prob[:, 1])
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"AP = {ap:.2f}",
                )
            )
        else:
        
            for i, cls in enumerate(classes):
                y_true_bin = (np.array(y_true) == cls).astype(int)
                y_prob_cls = y_prob[:, i]
                precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_cls)
                ap = average_precision_score(y_true_bin, y_prob_cls)
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode="lines",
                        name=f"Class {cls} AP = {ap:.2f}",
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1]),
            template="plotly_white",
        )

        if path:
            fig.write_image(path)
        else:
            fig.show()


def generate_regression_calibration_plot(y_true, y_pred, num_bins=10, title="Regression Calibration Plot", path=None):
    
    """Generate a calibration plot for regression"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    bins = np.array_split(np.arange(len(y_pred_sorted)), num_bins)
    
    bin_means_pred = [y_pred_sorted[bin_indices].mean() for bin_indices in bins]
    bin_means_true = [y_true_sorted[bin_indices].mean() for bin_indices in bins]
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_means_pred, bin_means_true, marker='o', linestyle='-', label='Binned Actual vs Predicted')
    plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], linestyle='--', color='gray', label='Ideal Calibration')
    plt.xlabel("Mean Predicted Value per bin")
    plt.ylabel("Mean Actual Value per bin")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generate_reg_error_distribution_plot(y_true, y_pred, title="Error Distribution", path=None):
    
    """Generate and save a plot showing the distribution of regression errors (residuals)."""
    errors = np.array(y_pred) - np.array(y_true)

    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, color='skyblue', bins=30)
    plt.axvline(0, color='red', line='--', label='Zero Error')
    plt.xlabel("Prediction Error (Residual)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()