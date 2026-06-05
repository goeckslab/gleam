import numpy as np
import pandas as pd
from plotly_plots import (
    _resolve_display_labels,
    build_prediction_diagnostics,
    build_binary_threshold_classification_plots_from_predictions,
    load_binary_threshold_data,
    optimize_binary_threshold_values,
)


def test_resolve_display_labels_decodes_ludwig_indices_before_friendly_lookup():
    assert _resolve_display_labels(["0", "1"], ["1", "2"]) == ["1", "2"]


def test_resolve_display_labels_preserves_original_numeric_labels():
    assert _resolve_display_labels(["1", "2"], ["1", "2"]) == ["1", "2"]


def test_optimize_binary_threshold_uses_requested_metric():
    result = optimize_binary_threshold_values(
        np.array([0, 1, 1, 0]),
        np.array([0.2, 0.4, 0.8, 0.7]),
        metric="f1",
    )

    assert result["threshold"] == 0.4
    assert result["metric_display"] == "F1"


def test_optimize_binary_threshold_tie_breaks_toward_half():
    result = optimize_binary_threshold_values(
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        metric="accuracy",
    )

    assert result["threshold"] == 0.5


def test_threshold_data_derives_positive_label_from_probability_suffix(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [1, 1, 1, 1],
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.1, 0.8, 0.2, 0.7],
            "label_probabilities_1": [0.9, 0.2, 0.8, 0.3],
        }
    ).to_csv(predictions_path, index=False)

    data = load_binary_threshold_data(str(predictions_path), split_value=1)

    assert data["positive_label"] == 1
    assert data["positive_probability_column"] == "label_probabilities_1"
    assert data["y_true_bin"].tolist() == [1, 0, 1, 0]
    assert data["y_score"].tolist() == [0.9, 0.2, 0.8, 0.3]


def test_binary_threshold_plots_are_recomputed_from_predictions(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [2, 2, 2, 2],
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.1, 0.8, 0.4, 0.7],
            "label_probabilities_1": [0.9, 0.2, 0.6, 0.3],
        }
    ).to_csv(predictions_path, index=False)

    plots = build_binary_threshold_classification_plots_from_predictions(
        str(predictions_path), threshold=0.5, split_value=2
    )

    assert [plot["title"] for plot in plots] == [
        "Confusion Matrix",
        "ROC Curve",
        "Precision-Recall Curve",
        "Per-Class metrics",
    ]
    assert "Selected Threshold: 0.500" in plots[0]["html"]


def test_binary_prediction_diagnostics_put_calibration_before_confidence(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [2, 2, 2, 2],
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.1, 0.8, 0.4, 0.7],
            "label_probabilities_1": [0.9, 0.2, 0.6, 0.3],
        }
    ).to_csv(predictions_path, index=False)

    plots = build_prediction_diagnostics(str(predictions_path), split_value=2)

    assert [plot["title"] for plot in plots] == [
        "Calibration Curve (Test)",
        "Prediction Confidence Distribution",
    ]
