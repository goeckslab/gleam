import numpy as np
from plotly_plots import _resolve_display_labels, optimize_binary_threshold_values


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
