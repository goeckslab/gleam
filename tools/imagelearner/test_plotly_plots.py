import numpy as np
import pandas as pd
import pytest
from plotly_plots import (
    _resolve_display_labels,
    _resolve_positive_label_from_probability,
    _select_positive_probability_column,
    build_binary_threshold_classification_plots_from_predictions,
    build_prediction_diagnostics,
    build_train_validation_plots,
    compute_binary_threshold_metrics_from_predictions,
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


def test_binary_threshold_plots_and_metrics_use_recorded_per_row_thresholds(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [2, 2, 2, 2],
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.1, 0.8, 0.45, 0.7],
            "label_probabilities_1": [0.9, 0.2, 0.55, 0.3],
            # fold 1 rows decide at 0.4, fold 2 rows at 0.6
            "decision_threshold": [0.4, 0.4, 0.6, 0.6],
        }
    ).to_csv(predictions_path, index=False)

    plots = build_binary_threshold_classification_plots_from_predictions(
        str(predictions_path), threshold=0.5, split_value=2
    )
    assert "Per-Fold Thresholds, Mean: 0.500" in plots[0]["html"]
    assert "Operating point (per-fold thresholds)" in plots[1]["html"]
    assert "Operating point (per-fold thresholds)" in plots[2]["html"]

    # row 3 is positive at the mean threshold but negative at its own fold's 0.6
    metrics = compute_binary_threshold_metrics_from_predictions(
        str(predictions_path), threshold=0.5, split_value=2
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.5)


def test_binary_threshold_plots_keep_single_threshold_when_rows_agree(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "split": [2, 2, 2, 2],
            "label": [1, 0, 1, 0],
            "label_probabilities_0": [0.1, 0.8, 0.4, 0.7],
            "label_probabilities_1": [0.9, 0.2, 0.6, 0.3],
            "decision_threshold": [0.5, 0.5, 0.5, 0.5],
        }
    ).to_csv(predictions_path, index=False)

    plots = build_binary_threshold_classification_plots_from_predictions(
        str(predictions_path), threshold=0.5, split_value=2
    )
    assert "Selected Threshold: 0.500" in plots[0]["html"]
    assert "Selected threshold=0.500" in plots[1]["html"]


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


@pytest.mark.parametrize("labels", [[1, 0], [0, 1], [1.0, 0.0], [True, False], ["0", "1"]])
@pytest.mark.parametrize("hint, expected", [("True", 1), ("False", 0), ("1", 1), ("0", 0)])
def test_positive_label_resolves_binary_aliases_independently_of_row_order(labels, hint, expected):
    label = _resolve_positive_label_from_probability(np.array(labels), hint)
    assert str(label).lower() in ({"true", "1", "1.0"} if expected else {"false", "0", "0.0"})


@pytest.mark.parametrize("labels", [["case", "control"], ["control", "case"]])
def test_positive_label_requires_unambiguous_mapping(labels):
    assert _resolve_positive_label_from_probability(np.array(labels), "True") is None
    assert _resolve_positive_label_from_probability(np.array(labels), None) is None
    assert _resolve_positive_label_from_probability(np.array(labels), "case") == "case"


@pytest.mark.parametrize("labels", [[0, 2], [0, -1], [0.0, 2.5], ["0", "2"], ["0", "-1"]])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_positive_label_resolves_zero_nonzero_numeric_labels(labels, reverse_rows):
    ordered = np.array(labels[::-1] if reverse_rows else labels)
    assert _resolve_positive_label_from_probability(ordered, "True") == labels[1]
    assert _resolve_positive_label_from_probability(ordered, "False") == labels[0]
    assert _resolve_positive_label_from_probability(ordered, str(labels[1])) == labels[1]


@pytest.mark.parametrize("labels", [[1, 2], [-1, 2], ["1", "2"], ["0", "0.0"], [0, np.nan], [0, 1, 2]])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_positive_label_rejects_non_bijective_numeric_mapping(labels, reverse_rows):
    ordered = np.array(labels[::-1] if reverse_rows else labels)
    for hint in ("True", "False"):
        assert _resolve_positive_label_from_probability(ordered, hint) is None


@pytest.mark.parametrize("positive_label", [2, -1])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_numeric_binary_labels_keep_threshold_optimization_and_diagnostics(tmp_path, positive_label, reverse_rows):
    predictions_path = tmp_path / "predictions.csv"
    frame = pd.DataFrame({
        "split": [1] * 4,
        "label": [0, positive_label, 0, positive_label],
        "label_probabilities_True": [0.1, 0.6, 0.55, 0.8],
    })
    if reverse_rows:
        frame = frame.iloc[::-1]
    frame.to_csv(predictions_path, index=False)
    data = load_binary_threshold_data(str(predictions_path), split_value=1)
    assert data["positive_label"] == positive_label
    assert data["negative_label"] == 0
    np.testing.assert_array_equal(data["y_true_bin"], frame["label"].ne(0))
    result = resolve_binary_threshold_for_report("auto", "f1", str(predictions_path), split_value=1)
    assert result["threshold"] == 0.6
    assert result["threshold_metric_value"] == 1.0
    assert result["threshold_source"] == "Optimized on validation split"
    metrics = compute_binary_threshold_metrics_from_predictions(str(predictions_path), 0.6, split_value=1)
    assert metrics["f1"] == metrics["accuracy"] == 1.0
    assert compute_binary_threshold_metrics_from_predictions(str(predictions_path), 0.5, split_value=1)["f1"] == 0.8
    plots = build_binary_threshold_classification_plots_from_predictions(str(predictions_path), 0.6, split_value=1)
    assert "ROC Curve (AUC=1.000)" in next(plot["html"] for plot in plots if plot["title"] == "ROC Curve")
    diagnostics = build_prediction_diagnostics(str(predictions_path), split_value=1)
    assert "Calibration Curve (Test)" in [plot["title"] for plot in diagnostics]


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [1, 0, 3, 2]])
def test_boolean_probability_exports_keep_threshold_metrics_and_auc_consistent(tmp_path, order):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame({
        "split": [1, 1, 1, 1],
        "label": [1, 0, 1, 0],
        "label_probabilities_False": [0.1, 0.8, 0.4, 0.7],
        "label_probabilities_True": [0.9, 0.2, 0.6, 0.3],
    }).iloc[order].to_csv(predictions_path, index=False)

    data = load_binary_threshold_data(str(predictions_path), split_value=1)
    assert data["positive_label"] == 1
    np.testing.assert_array_equal(data["y_true_bin"], np.array([1, 0, 1, 0])[order])
    result = resolve_binary_threshold_for_report(
        "auto", "balanced_accuracy", str(predictions_path), split_value=1
    )
    assert result["threshold"] == 0.5
    assert result["threshold_metric_value"] == 1.0
    metrics = compute_binary_threshold_metrics_from_predictions(
        str(predictions_path), result["threshold"], split_value=1
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    plots = build_binary_threshold_classification_plots_from_predictions(
        str(predictions_path), threshold=result["threshold"], split_value=1
    )
    roc_plot = next(plot for plot in plots if plot["title"] == "ROC Curve")
    assert "ROC Curve (AUC=1.000)" in roc_plot["html"]


@pytest.mark.parametrize("reverse_rows", [False, True])
@pytest.mark.parametrize("threshold, expected", [
    (0.5, [[2, 2], [1, 3]]),
    (0.75, [[2, 2], [3, 1]]),
])
def test_binary_confusion_matrix_counts_labels_and_annotations(
    tmp_path, monkeypatch, reverse_rows, threshold, expected
):
    import plotly_plots

    predictions_path = tmp_path / "predictions.csv"
    frame = pd.DataFrame({
        "split": [2] * 8,
        "label": [1, 0, 1, 0, 0, 1, 1, 0],
        "label_probabilities_True": [0.9, 0.8, 0.6, 0.2, 0.1, 0.7, 0.4, 0.8],
    })
    frame["label_probabilities_False"] = 1 - frame["label_probabilities_True"]
    if reverse_rows:
        frame = frame.iloc[::-1]
    frame.to_csv(predictions_path, index=False)
    figures = {}

    def capture_plot(title, fig, **kwargs):
        figures[title] = fig
        return {"title": title, "html": ""}

    monkeypatch.setattr(plotly_plots, "_wrap_plot", capture_plot)
    build_binary_threshold_classification_plots_from_predictions(
        str(predictions_path), threshold=threshold, split_value=2
    )
    fig = figures["Confusion Matrix"]
    np.testing.assert_array_equal(fig.data[0].z, expected)
    assert list(fig.data[0].x) == ["0", "1"]
    assert list(fig.data[0].y) == ["0", "1"]
    assert fig.layout.xaxis.title.text == "Predicted"
    assert fig.layout.yaxis.title.text == "Observed"
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.type == "category"
        assert axis.categoryorder == "array"
        assert list(axis.categoryarray) == ["0", "1"]
    assert fig.layout.yaxis.autorange == "reversed"
    assert f"Selected Threshold: {threshold:.3f}" in fig.layout.title.text
    annotations = {(a.y, a.x): a.text for a in fig.layout.annotations}
    assert len(annotations) == 4
    for i in range(2):
        for j in range(2):
            count = expected[i][j]
            assert annotations[str(i), str(j)] == f"<b>{count}</b><br>{count / 8 * 100:.1f}%"
    metrics = compute_binary_threshold_metrics_from_predictions(
        str(predictions_path), threshold=threshold, split_value=2
    )
    tn, fp = expected[0]
    fn, tp = expected[1]
    assert metrics["accuracy"] == (tn + tp) / 8
    assert metrics["recall"] == tp / (tp + fn)
    assert metrics["specificity"] == tn / (tn + fp)


def test_statistics_confusion_matrix_keeps_numeric_class_axes_categorical(tmp_path, monkeypatch):
    import plotly_plots

    stats_path = tmp_path / "test_statistics.json"
    stats_path.write_text('{"label": {"confusion_matrix": [[2, 2], [1, 3]]}}')
    figures = {}

    def capture_plot(title, fig, **kwargs):
        figures[title] = fig
        return {"title": title, "html": ""}

    monkeypatch.setattr(plotly_plots, "_wrap_plot", capture_plot)
    plotly_plots.build_classification_plots(str(stats_path))
    fig = figures["Confusion Matrix"]
    np.testing.assert_array_equal(fig.data[0].z, [[2, 2], [1, 3]])
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.type == "category"
        assert axis.categoryorder == "array"
        assert list(axis.categoryarray) == list(fig.data[0].x)
    assert fig.layout.yaxis.autorange == "reversed"


@pytest.mark.parametrize(
    "suffixes, expected",
    [(["0", "-2"], "-2"), (["0", "2"], "2"), (["0", "-1"], "-1"), (["False", "True"], "True")],
)
def test_positive_probability_column_keeps_ludwig_nonzero_class(suffixes, expected):
    columns = [f"label_probabilities_{suffix}" for suffix in suffixes]
    assert _select_positive_probability_column(columns) == (f"label_probabilities_{expected}", expected)


@pytest.mark.parametrize(
    "names",
    [
        ["0", "2"], ["False", "True"], ["event", "no_event"], ["negative", "positive"],
        ["benign", "malignant"], ["class1", "class2"], ["HER2-", "HER2+"], ["no", "yes"],
    ],
)
def test_report_and_ensemble_pick_the_same_positive_class(names):
    from ensemble import positive_class_index

    for ordered in (names, names[::-1]):
        columns = [f"label_probabilities_{name}" for name in ordered]
        assert _select_positive_probability_column(columns)[1] == ordered[positive_class_index(ordered)]


def test_positive_label_matches_numeric_hint_against_float_labels():
    assert _resolve_positive_label_from_probability(np.array([0.0, -2.0]), "-2") == -2.0
    assert _resolve_positive_label_from_probability(np.array([2.0, 0.0]), "2") == 2.0


def test_resolve_display_labels_keeps_integer_class_names_in_ludwig_order():
    # Ludwig keys per-class stats by class name; idx2str is frequency-ordered.
    assert _resolve_display_labels(["0", "1", "2"], ["2", "0", "1"]) == ["0", "1", "2"]


def test_resolve_display_labels_matches_canonical_ensemble_names():
    # Ensemble statistics use "1" where Ludwig's metadata says "1.0".
    assert _resolve_display_labels(["1", "2", "3"], ["1.0", "2.0", "3.0"]) == ["1.0", "2.0", "3.0"]
    assert _resolve_display_labels(["3", "1", "2"], ["1.0", "2.0", "3.0"]) == ["3.0", "1.0", "2.0"]
    assert _resolve_display_labels(["cat", "dog"], [" cat", "dog "]) == [" cat", "dog "]
