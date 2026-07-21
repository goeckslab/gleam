import numpy as np
import pandas as pd
from plotly_plots import (
    _resolve_display_labels,
    build_binary_threshold_classification_plots_from_predictions,
    build_prediction_diagnostics,
    build_train_validation_plots,
    load_binary_threshold_data,
    optimize_binary_threshold_values,
    resolve_binary_threshold_for_report,
)


def test_resolve_display_labels_decodes_ludwig_indices_before_friendly_lookup():
    assert _resolve_display_labels(["0", "1"], ["1", "2"]) == ["1", "2"]


def test_resolve_display_labels_preserves_original_numeric_labels():
    assert _resolve_display_labels(["1", "2"], ["1", "2"]) == ["1", "2"]


def test_train_validation_plots_include_hits_at_3_when_available(tmp_path):
    stats_path = tmp_path / "training_statistics.json"
    stats_path.write_text(
        """
{
  "training": {"label": {"hits_at_k": [0.2, 0.4], "loss": [1.2, 1.0]}},
  "validation": {"label": {"hits_at_k": [0.1, 0.3], "loss": [1.4, 1.1]}}
}
"""
    )

    plots = build_train_validation_plots(str(stats_path), top_k=3)
    titles = [plot["title"] for plot in plots]

    assert "Hits@3 across epochs (correct class in top 3)" in titles


def _write_multiclass_training_stats(tmp_path):
    stats_path = tmp_path / "training_statistics.json"
    stats_path.write_text(
        """
{
  "training": {"label": {"accuracy": [0.8, 0.98], "accuracy_micro": [0.82, 0.98],
                         "roc_auc": [0.95, 0.99], "loss": [1.2, 0.05]}},
  "validation": {"label": {"accuracy": [0.7, 0.87], "accuracy_micro": [0.72, 0.87],
                           "roc_auc": [0.93, 0.98], "loss": [1.4, 0.33]}}
}
"""
    )
    return stats_path


def test_train_validation_plot_titles_use_multiclass_metric_labels(tmp_path):
    """Curve titles must match the performance tables for category runs.

    Ludwig's multiclass "accuracy" is macro-averaged per-class recall and its
    "roc_auc" is macro-averaged, so plotting them as plain "Accuracy"/"ROC-AUC"
    contradicts the summary tables.
    """
    stats_path = _write_multiclass_training_stats(tmp_path)

    titles = [
        plot["title"]
        for plot in build_train_validation_plots(
            str(stats_path), top_k=3, output_type="category"
        )
    ]

    assert "Balanced Accuracy (Macro Recall) across epochs" in titles
    assert "Accuracy across epochs" in titles  # from accuracy_micro
    assert "Macro ROC-AUC across epochs" in titles
    assert "Overfitting gap: Macro ROC-AUC across epochs" in titles
    assert "Micro Accuracy across epochs" not in titles
    assert "ROC-AUC across epochs" not in titles
    assert "Overfitting gap: ROC-AUC across epochs" not in titles


def test_train_validation_plot_titles_unchanged_for_binary_and_default(tmp_path):
    stats_path = _write_multiclass_training_stats(tmp_path)

    for output_type in ("binary", None):
        titles = [
            plot["title"]
            for plot in build_train_validation_plots(
                str(stats_path), top_k=3, output_type=output_type
            )
        ]
        assert "Accuracy across epochs" in titles
        assert "ROC-AUC across epochs" in titles
        assert "Overfitting gap: ROC-AUC across epochs" in titles
        assert "Balanced Accuracy (Macro Recall) across epochs" not in titles
        assert "Macro ROC-AUC across epochs" not in titles


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


def test_report_threshold_preserves_training_selected_threshold_when_recompute_unavailable(tmp_path):
    result = resolve_binary_threshold_for_report(
        threshold_mode="auto",
        requested_metric="f1",
        predictions_path=str(tmp_path / "missing_predictions.csv"),
        existing_threshold=0.27,
    )

    assert result["threshold"] == 0.27
    assert result["threshold_source"] == (
        "Selected during training "
        "(report validation probabilities unavailable)"
    )
    assert result["threshold_metric"] == "F1"
    assert result["threshold_source"] != (
        "Default 0.5 (automatic optimization unavailable)"
    )


def test_report_threshold_recomputes_from_validation_predictions(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [1, 1, 1, 1],
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.6, 0.8, 0.2, 0.3],
            "label_probabilities_1": [0.4, 0.2, 0.8, 0.7],
        }
    ).to_csv(predictions_path, index=False)

    result = resolve_binary_threshold_for_report(
        threshold_mode="auto",
        requested_metric="f1",
        predictions_path=str(predictions_path),
        existing_threshold=0.27,
    )

    assert result["threshold"] == 0.4
    assert result["threshold_source"] == "Optimized on validation split"
    assert result["threshold_metric"] == "F1"
    assert result["threshold_metric_value"] == 0.8


def test_report_threshold_does_not_optimize_on_test_only_predictions(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    label_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.1, 0.8, 0.4, 0.7],
            "label_probabilities_1": [0.9, 0.2, 0.6, 0.3],
        }
    ).to_csv(predictions_path, index=False)
    pd.DataFrame(
        {
            "split": [1, 1, 2, 2, 2, 2],
            "label": [1, 0, 1, 0, 1, 0],
        }
    ).to_csv(label_path, index=False)

    result = resolve_binary_threshold_for_report(
        threshold_mode="auto",
        requested_metric="f1",
        predictions_path=str(predictions_path),
        label_data_path=str(label_path),
    )

    assert result["threshold"] == 0.5
    assert result["threshold_source"] == (
        "Default 0.5 (automatic optimization unavailable)"
    )


def test_threshold_data_accepts_validation_prediction_file_without_split_column(tmp_path):
    predictions_path = tmp_path / "validation_predictions.csv"
    label_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "label_probabilities_0": [0.6, 0.8],
            "label_probabilities_1": [0.4, 0.2],
        }
    ).to_csv(predictions_path, index=False)
    pd.DataFrame(
        {
            "split": [1, 1, 2, 2],
            "label": [1, 0, 1, 0],
        }
    ).to_csv(label_path, index=False)

    data = load_binary_threshold_data(
        str(predictions_path), label_data_path=str(label_path), split_value=1
    )

    assert data["y_true_bin"].tolist() == [1, 0]
    assert data["y_score"].tolist() == [0.4, 0.2]


def test_report_threshold_uses_default_only_without_training_threshold_or_predictions(tmp_path):
    result = resolve_binary_threshold_for_report(
        threshold_mode="auto",
        requested_metric="f1",
        predictions_path=str(tmp_path / "missing_predictions.csv"),
    )

    assert result["threshold"] == 0.5
    assert result["threshold_source"] == (
        "Default 0.5 (automatic optimization unavailable)"
    )
    assert result["threshold_metric"] == "F1"


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


def test_prediction_diagnostics_skip_calibration_for_multiclass_labels(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [2, 2, 2, 2],
            "label": ["a", "b", "c", "a"],
            "label_probability": [0.7, 0.8, 0.6, 0.9],
        }
    ).to_csv(predictions_path, index=False)

    plots = build_prediction_diagnostics(str(predictions_path), split_value=2)

    assert [plot["title"] for plot in plots] == [
        "Prediction Confidence Distribution",
    ]


def test_prediction_diagnostics_handles_probability_export_without_labels(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [2, 2],
            "probabilities": ["[0.2, 0.8]", "[0.7, 0.3]"],
        }
    ).to_csv(predictions_path, index=False)

    plots = build_prediction_diagnostics(str(predictions_path), split_value=2)

    assert [plot["title"] for plot in plots] == [
        "Prediction Confidence Distribution",
    ]
