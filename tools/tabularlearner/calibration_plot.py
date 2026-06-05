"""Calibration curve plotting utilities for binary classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve


ArrayLike = Sequence[float] | np.ndarray
ModelCalibrationData = Mapping[str, Mapping[str, ArrayLike]]


def _validate_binary_calibration_inputs(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional binary labels and probabilities."""
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError(
            "y_true and y_prob must have the same length; "
            f"got {y_true_arr.shape[0]} and {y_prob_arr.shape[0]}."
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_prob must contain at least one sample.")

    if not np.isfinite(y_prob_arr).all():
        raise ValueError("y_prob must contain only finite probability values.")

    labels = set(np.unique(y_true_arr).tolist())
    if not labels.issubset({0, 1, False, True}):
        raise ValueError("y_true must contain only binary labels: 0 or 1.")

    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("y_prob values must be between 0 and 1.")

    return y_true_arr.astype(int), y_prob_arr


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> float:
    """
    Calculate expected calibration error for binary probabilities.

    ECE is the weighted mean absolute difference between each bin's observed
    positive-label fraction and mean predicted probability.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    y_true_arr, y_prob_arr = _validate_binary_calibration_inputs(y_true, y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bin_edges[1:-1], right=True)

    ece = 0.0
    total = y_prob_arr.shape[0]
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        bin_accuracy = float(y_true_arr[mask].mean())
        bin_confidence = float(y_prob_arr[mask].mean())
        ece += (bin_count / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_calibration_curves(
    models: ModelCalibrationData,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Calibration Curve",
    save_path: Optional[str | Path] = None,
) -> tuple[Figure, Axes]:
    """
    Plot calibration curves for one or more binary classification models.

    Parameters
    ----------
    models
        Mapping of model names to ``{"y_true": ..., "y_prob": ...}`` arrays.
    n_bins
        Number of bins used by ``sklearn.calibration.calibration_curve``.
    strategy
        Binning strategy. Supported values are ``"uniform"`` and ``"quantile"``.
    title
        Plot title.
    save_path
        Optional output path. When provided, the figure is saved with
        ``bbox_inches="tight"``.
    """
    if not models:
        raise ValueError("models must contain at least one model.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError('strategy must be either "uniform" or "quantile".')

    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, values in models.items():
        if "y_true" not in values or "y_prob" not in values:
            raise ValueError(
                f"{model_name!r} must provide both 'y_true' and 'y_prob'."
            )
        y_true_arr, y_prob_arr = _validate_binary_calibration_inputs(
            values["y_true"],
            values["y_prob"],
        )
        prob_true, prob_pred = calibration_curve(
            y_true_arr,
            y_prob_arr,
            n_bins=n_bins,
            strategy=strategy,
        )
        ece = expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins)
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"{model_name} - ECE: {ece:.3f}",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Perfect calibration",
    )
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    models = {
        "Example model": {
            "y_true": np.array([0, 0, 1, 0, 1, 1, 1, 1]),
            "y_prob": np.array([0.05, 0.12, 0.18, 0.35, 0.42, 0.78, 0.83, 0.91]),
        }
    }

    fig, ax = plot_calibration_curves(
        models=models,
        n_bins=5,
        strategy="uniform",
        title="Model Calibration Curve",
        save_path="calibration_curve.png",
    )

    plt.show()
