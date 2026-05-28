import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


def labels_in_sample_order(*values):
    """Return labels in first-seen sample order across one or more arrays."""
    series = [
        pd.Series(value).reset_index(drop=True)
        for value in values
        if value is not None
    ]
    if not series:
        return []
    return pd.unique(pd.concat(series, ignore_index=True)).tolist()


def labels_in_metric_order(values):
    """Return labels in deterministic metric order when probability order is unknown."""
    labels = labels_in_sample_order(values)
    try:
        return sorted(labels)
    except Exception:
        return labels


def positive_class_label(labels):
    labels = [] if labels is None else list(labels)
    if not labels:
        return 1
    return 1 if 1 in labels else labels[-1]


def probability_class_labels(y_true, y_score, model_classes=None):
    """
    Resolve the class label represented by each probability column.

    scikit-learn probability columns follow estimator.classes_. If that is
    unavailable, fall back to first-seen labels only when the count matches.
    """
    scores = np.asarray(y_score)
    classes = [] if model_classes is None else list(model_classes)
    if scores.ndim == 2 and classes and len(classes) == scores.shape[1]:
        return classes

    sample_labels = labels_in_sample_order(y_true)
    if scores.ndim == 2 and len(sample_labels) == scores.shape[1]:
        return sample_labels
    if scores.ndim == 1 and len(classes) >= 2:
        return classes
    return classes or sample_labels


def weighted_ovr_pr_auc(y_true, y_score, labels=None, pos_label=None):
    """
    Compute curve-based PR-AUC.

    Binary tasks use one positive-class curve. Multiclass tasks use one-vs-rest
    curves in probability-column order and return a support-weighted mean.
    """
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    class_labels = (
        list(labels) if labels is not None else labels_in_metric_order(y_true_series)
    )
    if len(class_labels) < 2:
        return np.nan

    scores = np.asarray(y_score)
    if len(scores) != len(y_true_series):
        return np.nan

    if len(class_labels) == 2:
        pos_label = (
            pos_label
            if pos_label is not None
            else positive_class_label(class_labels)
        )
        if scores.ndim == 2:
            if scores.shape[1] < 2:
                scores = scores.ravel()
            else:
                try:
                    pos_idx = class_labels.index(pos_label)
                except ValueError:
                    pos_idx = scores.shape[1] - 1
                scores = scores[:, min(pos_idx, scores.shape[1] - 1)]
        precision, recall, _ = precision_recall_curve(
            (y_true_series == pos_label).astype(int),
            scores,
        )
        return auc(recall, precision)

    if scores.ndim != 2 or scores.shape[1] != len(class_labels):
        return np.nan

    weighted_total = 0.0
    support_total = 0
    for class_idx, class_label in enumerate(class_labels):
        y_true_bin = (y_true_series == class_label).astype(int)
        if len(pd.unique(y_true_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(
            y_true_bin,
            scores[:, class_idx],
        )
        support = int(y_true_bin.sum())
        weighted_total += auc(recall, precision) * support
        support_total += support

    return weighted_total / support_total if support_total else np.nan
