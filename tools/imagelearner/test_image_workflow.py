import importlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pandas as pd


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

if "ludwig" not in sys.modules:
    ludwig = types.ModuleType("ludwig")
    ludwig_globals = types.ModuleType("ludwig.globals")
    ludwig_globals.PREDICTIONS_PARQUET_FILE_NAME = "predictions.parquet"
    ludwig.globals = ludwig_globals
    sys.modules["ludwig"] = ludwig
    sys.modules["ludwig.globals"] = ludwig_globals

if "ludwig_backend" not in sys.modules:
    ludwig_backend = types.ModuleType("ludwig_backend")

    class Backend:
        pass

    ludwig_backend.Backend = Backend
    sys.modules["ludwig_backend"] = ludwig_backend


def _load_test_objects():
    constants = importlib.import_module("constants")
    html_structure = importlib.import_module("html_structure")
    image_workflow = importlib.import_module("image_workflow")
    return (
        constants.IMAGE_PATH_COLUMN_NAME,
        constants.LABEL_COLUMN_NAME,
        html_structure.format_image_match_notice,
        image_workflow.ImageLearnerCLI,
    )


def _config_table_value_html(html: str, label: str) -> str:
    match = re.search(
        r"<tr>\s*<td\b[^>]*>\s*"
        + re.escape(label)
        + r"\s*</td>\s*<td\b[^>]*>(?P<value>.*?)</td>\s*</tr>",
        html,
        flags=re.DOTALL,
    )
    assert match, f"{label} row not found in config table"
    return match.group("value")


def test_map_image_paths_filters_to_matching_images_and_records_summary(tmp_path):
    test_objects = _load_test_objects()
    image_path_col = test_objects[0]
    label_col = test_objects[1]
    ImageLearnerCLI = test_objects[3]
    image_dir = tmp_path / "images"
    nested_dir = image_dir / "nested"
    nested_dir.mkdir(parents=True)
    (image_dir / "case_a.jpg").write_bytes(b"image-a")
    (nested_dir / "case_b.png").write_bytes(b"image-b")
    (image_dir / "unused.webp").write_bytes(b"unused")
    (image_dir / "notes.txt").write_text("not an image")

    df = pd.DataFrame(
        {
            image_path_col: ["case_a.jpg", "case_b", "missing.png"],
            label_col: ["akiec", "bcc", "akiec"],
        }
    )
    cli = object.__new__(ImageLearnerCLI)
    cli.image_extract_dir = image_dir
    cli.image_match_summary = {}

    filtered = ImageLearnerCLI._map_image_paths_with_search(cli, df)

    assert filtered[image_path_col].tolist() == [
        "images/case_a.jpg",
        "images/nested/case_b.png",
    ]
    assert filtered[label_col].tolist() == ["akiec", "bcc"]
    assert cli.image_match_summary == {
        "csv_rows_total": 3,
        "matched_rows": 2,
        "csv_rows_missing_images": 1,
        "zip_images_total": 3,
        "zip_images_missing_csv_rows": 1,
    }


def test_format_image_match_notice_summarizes_matching_sample_count():
    format_image_match_notice = _load_test_objects()[2]
    html = format_image_match_notice(
        {
            "csv_rows_total": 10,
            "matched_rows": 8,
            "csv_rows_missing_images": 2,
            "zip_images_total": 9,
            "zip_images_missing_csv_rows": 1,
        }
    )

    assert "CSV/ZIP Matching Notice" in html
    assert "Mismatches were detected between the CSV and ZIP files." in html
    assert "The final experiment utilized 8 fully matched samples." in html
    assert "metadata row(s) were excluded" not in html
    assert "extracted image file(s) were not used" not in html


def test_format_image_match_notice_omits_clean_matches():
    format_image_match_notice = _load_test_objects()[2]
    assert (
        format_image_match_notice(
            {
                "csv_rows_total": 2,
                "matched_rows": 2,
                "csv_rows_missing_images": 0,
                "zip_images_total": 2,
                "zip_images_missing_csv_rows": 0,
            }
        )
        == ""
    )


def test_metric_summary_formats_hits_at_3_without_changing_metric_values():
    html_structure = importlib.import_module("html_structure")
    train_stats = {
        "training": {
            "label": {
                "accuracy": [0.2],
                "hits_at_k": [0.8],
                "loss": [1.25],
                "roc_auc": [0.7],
            }
        },
        "validation": {
            "label": {
                "accuracy": [0.3],
                "hits_at_k": [0.6],
                "loss": [1.5],
                "roc_auc": [0.75],
            }
        },
    }
    test_stats = {
        "label": {
            "accuracy": 0.4,
            "hits_at_k": 0.9,
            "loss": 1.1,
            "roc_auc": 0.8,
            "per_class_stats": {"a": {}, "b": {}, "c": {}},
        }
    }

    html = html_structure.format_stats_table_html(
        train_stats, test_stats, "category", top_k=3
    )

    assert "Hits@3" in html
    assert "0.9000" in html
    assert "90.0%" not in html
    assert "Loss and regression errors remain raw scores" not in html
    assert "1.1000" in html


def test_config_table_validation_metric_label_matches_multiclass_tables():
    """Config table must use the same category-aware labels as the metric tables.

    Ludwig's multiclass "accuracy" is macro-averaged per-class recall, so a run
    validated on it has to read "Balanced Accuracy (Macro Recall)" in the config
    table too - otherwise it contradicts the performance summaries.
    """
    html_structure = importlib.import_module("html_structure")

    multiclass_html = html_structure.format_config_table_html(
        {"validation_metric": "accuracy"}, output_type="category"
    )
    value_html = _config_table_value_html(multiclass_html, "Validation Metric")
    assert value_html == "Balanced Accuracy (Macro Recall)"

    # The label must agree with the one used in the performance tables.
    assert value_html == html_structure.format_metric_display_name(
        "accuracy", output_type="category"
    )

    # accuracy_micro is the true sample-level accuracy for multiclass.
    micro_html = html_structure.format_config_table_html(
        {"validation_metric": "accuracy_micro"}, output_type="category"
    )
    assert _config_table_value_html(micro_html, "Validation Metric") == "Accuracy"

    # Binary and unknown output types keep the original labels.
    binary_html = html_structure.format_config_table_html(
        {"validation_metric": "accuracy"}, output_type="binary"
    )
    assert _config_table_value_html(binary_html, "Validation Metric") == "Accuracy"

    default_html = html_structure.format_config_table_html(
        {"validation_metric": "accuracy"}
    )
    assert _config_table_value_html(default_html, "Validation Metric") == "Accuracy"

    # top_k resolution still works alongside the new argument.
    hits_html = html_structure.format_config_table_html(
        {"validation_metric": "hits_at_k", "top_k": 5}, output_type="category"
    )
    assert _config_table_value_html(hits_html, "Validation Metric") == "Hits@5"
    assert _config_table_value_html(hits_html, "Hits@K") == "5"
    assert "Hits@5 Top Classes" not in hits_html
    assert "the correct class must appear" not in hits_html


def test_runtime_resource_metadata_reports_visible_gpus(monkeypatch):
    ImageLearnerCLI = _load_test_objects()[3]
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert ImageLearnerCLI._runtime_resource_metadata(3) == {
        "compute_resource": "GPU (2 visible CUDA devices)",
        "preprocessing_worker_count": 3,
    }


def test_runtime_resource_metadata_reports_galaxy_cpu_allocation(monkeypatch):
    ImageLearnerCLI = _load_test_objects()[3]
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("GALAXY_SLOTS", "8")

    assert ImageLearnerCLI._runtime_resource_metadata(4) == {
        "compute_resource": "CPU (8 process-available cores)",
        "preprocessing_worker_count": 4,
    }


def test_runtime_resource_metadata_falls_back_to_cpu_affinity(monkeypatch):
    ImageLearnerCLI = _load_test_objects()[3]
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("GALAXY_SLOTS", "invalid")
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2},
        raising=False,
    )

    assert ImageLearnerCLI._runtime_resource_metadata(1) == {
        "compute_resource": "CPU (3 process-available cores)",
        "preprocessing_worker_count": 1,
    }


def test_config_table_formats_resource_rows_and_omits_legacy_missing_rows():
    html_structure = importlib.import_module("html_structure")
    html = html_structure.format_config_table_html(
        {
            "compute_resource": "GPU <2 visible devices>",
            "preprocessing_worker_count": 5,
        }
    )

    assert _config_table_value_html(
        html, "Compute Resource"
    ) == "GPU &lt;2 visible devices&gt;"
    assert _config_table_value_html(
        html, "Preprocessing Worker Count"
    ) == "5"

    legacy_html = html_structure.format_config_table_html(
        {"architecture": "ResNet"}
    )
    assert "Compute Resource" not in legacy_html
    assert "Preprocessing Worker Count" not in legacy_html


def test_config_table_shows_model_compatible_image_size_when_adapted():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {
            "image_size": "96x96",
            "image_size_adaptation": {
                "original_size": "96x96",
                "requested_resize": "original",
                "training_size": "96x96",
                "final_training_size": "96x96",
                "model_configured_size": "384x384",
                "model_adaptation_size": "224x224",
                "model_adaptation_from_size": "96x96",
                "model_adaptation_to_size": "224x224",
            },
        }
    )
    image_size_html = _config_table_value_html(html, "Image Size")

    assert "224x224" in image_size_html
    assert "text-align: center" in image_size_html
    assert "font-size: 0.85em" in image_size_html
    assert "Auto-resize for model compatibility" in image_size_html
    assert image_size_html.index(
        "Auto-resize for model compatibility"
    ) < image_size_html.index("224x224")
    assert "Image was resized to be compatible with the model selected" not in image_size_html
    assert "Resized for model compatibility" not in image_size_html
    assert "Original image size" not in image_size_html
    assert "Final resize before training" not in image_size_html
    assert "Model original input size" not in image_size_html
    assert "Model adaptation" not in image_size_html
    assert "Training preprocessing size" not in image_size_html
    assert "Model configured input size" not in image_size_html
    assert "Auto-adaptation for the model" not in image_size_html


def test_config_table_shows_user_selected_image_size_without_adaptation():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {
            "image_size": "384x384",
            "image_size_adaptation": {
                "original_size": "mixed",
                "original_sizes": [
                    {"size": "96x96", "count": 2},
                    {"size": "128x128", "count": 1},
                ],
                "is_mixed": True,
                "requested_resize": "384x384",
                "training_size": "384x384",
                "final_training_size": "384x384",
            },
        }
    )
    image_size_html = _config_table_value_html(html, "Image Size")

    assert "384x384" in image_size_html
    assert "Resized for model compatibility" not in image_size_html
    assert "Auto-resize for model compatibility" not in image_size_html
    assert "Image was resized to be compatible with the model selected" not in image_size_html
    assert "font-size: 0.85em" not in image_size_html
    assert "text-align: center" in image_size_html
    assert "Original image sizes" not in image_size_html
    assert "Final resize before training" not in image_size_html
    assert "Training preprocessing size" not in image_size_html


def test_image_size_details_are_written_to_artifacts(tmp_path):
    (_image_path_col, _label_col, _notice, ImageLearnerCLI) = _load_test_objects()
    cli = ImageLearnerCLI.__new__(ImageLearnerCLI)
    exp_dir = tmp_path / "experiment_run"
    exp_dir.mkdir()
    details = {
        "original_size": "96x96",
        "final_training_size": "96x96",
        "model_configured_size": "384x384",
        "model_adaptation_to_size": "224x224",
    }

    cli._write_image_size_details_artifact(
        tmp_path,
        {"image_size_adaptation": details},
    )

    artifact_path = exp_dir / "image_size_details.json"
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text()) == details


def test_config_table_displays_resolved_auto_batch_size():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {"batch_size": "auto"},
        training_progress={"batch_size": 16},
    )
    batch_size_html = _config_table_value_html(html, "Batch Size")

    assert batch_size_html == (
        "<span style='font-size: 0.85em;'>"
        "Auto-selection</span><br>"
        "16"
    )
    assert "<span style='font-size: 0.85em;'>16</span>" not in batch_size_html
    assert "Based on model architecture and training setup" not in batch_size_html
    assert "Ludwig Trainer Parameters" not in batch_size_html
    assert "Auto-selected batch size by Ludwig:" not in batch_size_html
    assert "Ludwig Auto-selection" not in batch_size_html
    assert "auto" not in batch_size_html


def test_config_table_displays_resolved_auto_learning_rate_without_extra_context():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {"learning_rate": "auto"},
        training_progress={"learning_rate": 1e-5},
    )
    learning_rate_html = _config_table_value_html(html, "Learning Rate")

    assert learning_rate_html == (
        "<span style='font-size: 0.85em;'>"
        "Auto-selection</span><br>"
        "1e-05"
    )
    assert "<span style='font-size: 0.85em;'>1e-05</span>" not in learning_rate_html
    assert "Based on model architecture and training setup" not in learning_rate_html
    assert "Auto-selected learning rate by Ludwig:" not in learning_rate_html
    assert "Ludwig Auto-selection" not in learning_rate_html


def test_config_table_displays_batch_learning_rate_and_image_size_together():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {
            "batch_size": "auto",
            "learning_rate": "auto",
            "image_size": "96x96",
            "image_size_adaptation": {
                "final_training_size": "96x96",
                "model_adaptation_to_size": "224x224",
            },
        },
        training_progress={"batch_size": 16, "learning_rate": 1e-5},
    )

    batch_size_html = _config_table_value_html(html, "Batch Size")
    learning_rate_html = _config_table_value_html(html, "Learning Rate")
    image_size_html = _config_table_value_html(html, "Image Size")

    assert batch_size_html == (
        "<span style='font-size: 0.85em;'>"
        "Auto-selection</span><br>"
        "16"
    )
    assert learning_rate_html == (
        "<span style='font-size: 0.85em;'>"
        "Auto-selection</span><br>"
        "1e-05"
    )
    assert "224x224" in image_size_html
    assert "Auto-resize for model compatibility" in image_size_html
    assert image_size_html.index(
        "Auto-resize for model compatibility"
    ) < image_size_html.index("224x224")
    assert "Auto-selected batch size by Ludwig:" not in batch_size_html
    assert "Auto-selected learning rate by Ludwig:" not in learning_rate_html
    assert "Ludwig Auto-selection" not in batch_size_html
    assert "Ludwig Auto-selection" not in learning_rate_html
    assert "Image was resized to be compatible with the model selected" not in image_size_html
