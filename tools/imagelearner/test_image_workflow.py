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

from constants import IMAGE_PATH_COLUMN_NAME, LABEL_COLUMN_NAME
from html_structure import format_image_match_notice
from image_workflow import ImageLearnerCLI


def test_map_image_paths_filters_to_matching_images_and_records_summary(tmp_path):
    image_dir = tmp_path / "images"
    nested_dir = image_dir / "nested"
    nested_dir.mkdir(parents=True)
    (image_dir / "case_a.jpg").write_bytes(b"image-a")
    (nested_dir / "case_b.png").write_bytes(b"image-b")
    (image_dir / "unused.webp").write_bytes(b"unused")
    (image_dir / "notes.txt").write_text("not an image")

    df = pd.DataFrame(
        {
            IMAGE_PATH_COLUMN_NAME: ["case_a.jpg", "case_b", "missing.png"],
            LABEL_COLUMN_NAME: ["akiec", "bcc", "akiec"],
        }
    )
    cli = object.__new__(ImageLearnerCLI)
    cli.image_extract_dir = image_dir
    cli.image_match_summary = {}

    filtered = ImageLearnerCLI._map_image_paths_with_search(cli, df)

    assert filtered[IMAGE_PATH_COLUMN_NAME].tolist() == [
        "images/case_a.jpg",
        "images/nested/case_b.png",
    ]
    assert filtered[LABEL_COLUMN_NAME].tolist() == ["akiec", "bcc"]
    assert cli.image_match_summary == {
        "csv_rows_total": 3,
        "matched_rows": 2,
        "csv_rows_missing_images": 1,
        "zip_images_total": 3,
        "zip_images_missing_csv_rows": 1,
    }


def test_format_image_match_notice_lists_both_mismatch_counts():
    html = format_image_match_notice(
        {
            "csv_rows_total": 10,
            "matched_rows": 8,
            "csv_rows_missing_images": 2,
            "zip_images_total": 9,
            "zip_images_missing_csv_rows": 1,
        }
    )

    assert "CSV/ZIP image matching notice" in html
    assert "2 metadata row(s) were excluded" in html
    assert "1 extracted image file(s) were not used" in html
    assert "The experiment used 8 matched row(s)" in html


def test_format_image_match_notice_omits_clean_matches():
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
