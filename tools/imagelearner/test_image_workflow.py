import importlib
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


def test_config_table_displays_resolved_auto_batch_size():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {"batch_size": "auto"},
        training_progress={"batch_size": 16},
    )

    assert "Auto-selected batch size by Ludwig" in html
    assert "<span style='font-size: 0.85em;'>16</span>" in html
    assert "Based on model architecture and training setup" not in html
    assert "Ludwig Trainer Parameters" not in html
    assert ">auto</td>" not in html


def test_config_table_displays_resolved_auto_learning_rate_without_extra_context():
    html_structure = importlib.import_module("html_structure")

    html = html_structure.format_config_table_html(
        {"learning_rate": "auto"},
        training_progress={"learning_rate": 1e-5},
    )

    assert "Auto-selected learning rate by Ludwig" in html
    assert "<span style='font-size: 0.85em;'>1e-05</span>" in html
    assert "Based on model architecture and training setup" not in html
