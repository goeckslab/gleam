from plotly_plots import _resolve_display_labels


def test_resolve_display_labels_decodes_ludwig_indices_before_friendly_lookup():
    assert _resolve_display_labels(["0", "1"], ["1", "2"]) == ["1", "2"]


def test_resolve_display_labels_preserves_original_numeric_labels():
    assert _resolve_display_labels(["1", "2"], ["1", "2"]) == ["1", "2"]
