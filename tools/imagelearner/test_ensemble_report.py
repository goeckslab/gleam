"""End-to-end check of the real HTML report over a fake ensemble run."""
import importlib
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

ludwig = types.ModuleType("ludwig")
ludwig.__path__ = []
ludwig_globals = types.ModuleType("ludwig.globals")
ludwig_globals.DESCRIPTION_FILE_NAME = "description.json"
ludwig_globals.PREDICTIONS_PARQUET_FILE_NAME = "predictions.parquet"
ludwig_globals.TEST_STATISTICS_FILE_NAME = "test_statistics.json"
ludwig_globals.TRAIN_SET_METADATA_FILE_NAME = "training_set_metadata.json"
ludwig_utils = types.ModuleType("ludwig.utils")
ludwig_data_utils = types.ModuleType("ludwig.utils.data_utils")
ludwig_data_utils.get_split_path = lambda dataset: f"{dataset}.split"
ludwig.globals = ludwig_globals
ludwig.utils = ludwig_utils
sys.modules["ludwig"] = ludwig
sys.modules["ludwig.globals"] = ludwig_globals
sys.modules["ludwig.utils"] = ludwig_utils
sys.modules["ludwig.utils.data_utils"] = ludwig_data_utils
sys.modules.pop("ludwig_backend", None)

ludwig_backend = importlib.import_module("ludwig_backend")
test_ensemble = importlib.import_module("test_ensemble")
image_workflow = importlib.import_module("image_workflow")
constants = importlib.import_module("constants")

SPLIT = constants.SPLIT_COLUMN_NAME
LABEL = constants.LABEL_COLUMN_NAME
IMAGE = constants.IMAGE_PATH_COLUMN_NAME


class ReportBackend(test_ensemble.FakeBackend, ludwig_backend.LudwigDirectBackend):
    """Fake training, real report."""

    def generate_html_report(self, title, output_dir, config, split_info):
        self.report_configs.append((dict(config), split_info, None))
        return ludwig_backend.LudwigDirectBackend.generate_html_report(
            self, title, output_dir, config, split_info
        )


class RegressionReportBackend(ReportBackend):
    def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
        df = pd.read_csv(dataset_path)
        exp_dir = Path(output_dir) / "experiment_run"
        (exp_dir / "model").mkdir(parents=True)
        (exp_dir / "model" / "model_weights").write_text("weights")
        (exp_dir / "description.json").write_text(json.dumps({"config": {"random_seed": random_seed}}))
        rng = np.random.RandomState(random_seed)
        test_df = df[df[SPLIT] == 2].reset_index(drop=True)
        pd.DataFrame(
            {"label_predictions": test_df[LABEL].to_numpy() + rng.normal(0, 0.5, len(test_df))}
        ).to_csv(exp_dir / "predictions.csv", index=False)
        (exp_dir / "test_statistics.json").write_text(
            json.dumps({"label": {"mean_squared_error": 0.3, "mean_absolute_error": 0.4, "r2": 0.9}})
        )
        (exp_dir / "training_statistics.json").write_text(
            json.dumps(
                {
                    "training": {"label": {"mean_absolute_error": [0.9, 0.5], "loss": [1.0, 0.4]}},
                    "validation": {"label": {"mean_absolute_error": [0.9, 0.6], "loss": [1.0, 0.5]}},
                }
            )
        )
        self.runs.append((Path(dataset_path).name, random_seed))

    def generate_split_predictions(self, output_dir, dataset_path, split_value, output_filename):
        df = pd.read_csv(dataset_path)
        rows = df[df[SPLIT] == split_value].reset_index(drop=True)
        if rows.empty:
            return None
        seed = next(seed for name, seed in reversed(self.runs) if name == Path(dataset_path).name)
        rng = np.random.RandomState(seed)
        out = rows[[SPLIT, LABEL, IMAGE]].copy()
        out["label_predictions"] = rows[LABEL].to_numpy() + rng.normal(0, 0.5, len(rows))
        path = Path(output_dir) / "experiment_run" / output_filename
        out.to_csv(path, index=False)
        return path


def test_report_renders_binary_ensemble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path, zip_path, _ = test_ensemble._make_dataset(tmp_path, n_per_class=16)
    args = test_ensemble._make_args(tmp_path, csv_path, zip_path, num_folds=2, num_seeds=2)
    backend = ReportBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    html = (tmp_path / "image_classification_results_report.html").read_text()
    assert "Ensemble Summary" in html
    assert "Member Models and Averaged Ensemble" in html
    assert "Member Validation and Test Metrics" in html
    assert "Members mean ± SD" in html
    assert "Ensemble (averaged)" in html
    for name in ("fold1_seed42", "fold1_seed43", "fold2_seed42", "fold2_seed43"):
        assert name in html
    # the Dataset Overview describes the folds, not a single split
    assert "Rows in folds" in html
    assert "Test (fixed)" in html
    assert "Rows per fold" in html
    assert "test = the fixed test split, averaged across all members" in html
    assert "Test Performance Summary" in html
    assert "fold1_seed42 (first trained member). It is one member, not the averaged ensemble" in html
    assert "Its own test metrics:" in html
    assert "Test rows scored" in html
    assert "scored the same way as the ensemble row" in html
    assert "Check before relying on these numbers" not in html
    assert "Ludwig Evaluation Threshold (Member Statistics Files)" in html
    assert "Selected Threshold (Test Summary)" in html
    assert "Source: Optimized on out-of-fold validation predictions (each row scored by its own fold&#x27;s models only" in html
    assert "selected-threshold test metrics unavailable" not in html
    assert "test rows are averaged across all members" in html
    assert "2-fold cross-validation (validation folds) x 2 random seeds" in html
    # binary plots are built from the averaged prediction CSVs
    assert "Threshold plot" in html
    assert "Confusion Matrix" in html


def test_report_renders_test_fold_ensemble_with_per_fold_thresholds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path, zip_path, base_df = test_ensemble._make_dataset(tmp_path, n_per_class=16, with_split=False)
    args = test_ensemble._make_args(
        tmp_path, csv_path, zip_path, fold_role="test", num_folds=2, num_seeds=1
    )
    backend = ReportBackend()
    recomputations = []

    def unexpected_recompute(*args, **kwargs):
        recomputations.append(args)
        raise AssertionError("ensemble metrics must retain the per-row thresholds")

    monkeypatch.setattr(ludwig_backend, "compute_binary_threshold_metrics_from_predictions", unexpected_recompute)

    image_workflow.ImageLearnerCLI(args, backend).run()
    assert not recomputations

    html = (tmp_path / "image_classification_results_report.html").read_text()
    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / "ensemble_summary.json").read_text())
    preds = pd.read_csv(exp_dir / "predictions.csv")
    assert len(preds) == len(base_df)
    # every row carries the threshold of the fold that held it out
    assert set(np.round(preds["decision_threshold"], 6)) == {
        round(v, 6) for v in summary["fold_thresholds"].values()
    }
    if len(set(summary["fold_thresholds"].values())) > 1:
        # plots re-derive labels from the per-row thresholds, not the mean
        assert "Per-Fold Thresholds, Mean:" in html
        assert "Operating point (per-fold thresholds)" in html
    assert "test = every row, scored out-of-fold" in html
    assert "Decision thresholds: per fold" in html
    assert "Mean of Per-Fold Test Thresholds" in html
    assert "Source: Ludwig model evaluation." not in html
    assert "selected-threshold test metrics unavailable" not in html
    # overview: per-fold layout, no fixed test column
    assert "Rows per fold" in html
    assert "Test (fixed)" not in html
    assert "held out as the test set" in html
    assert sum(row["test"] for row in summary["fold_layout"]) == len(base_df)
    assert all(
        row["train"] + row["validation"] + row["test"] == len(base_df)
        for row in summary["fold_layout"]
    )


def test_report_renders_regression_ensemble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = []
    for idx in range(40):
        rows.append(
            {
                IMAGE: f"images/img_{idx}.png",
                LABEL: 10.0 + idx * 0.37,
                SPLIT: 2 if idx % 4 == 3 else (1 if idx % 4 == 2 else 0),
            }
        )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in df[IMAGE]:
            zf.writestr(name, b"not really a png")
    args = test_ensemble._make_args(
        tmp_path, csv_path, zip_path, sample_id_column=None, cross_validation=False, num_seeds=3
    )
    backend = RegressionReportBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / "ensemble_summary.json").read_text())
    assert summary["task_type"] == "regression"
    assert summary["aggregation"] == "mean of predicted values"
    assert "root_mean_squared_error" in summary["ensemble_test_metrics"]
    html = (tmp_path / "image_classification_results_report.html").read_text()
    assert "Ensemble Summary" in html
    assert "Ground Truth vs. Predictions" in html
    assert "seed44" in html


def test_ensemble_summary_flags_gaps_and_default_fold_thresholds():
    html_structure = importlib.import_module("html_structure")
    summary = {
        "cross_validation": True,
        "num_folds": 2,
        "fold_role": "test",
        "seeds": [42],
        "members_trained": 2,
        "members_planned": 2,
        "threshold": 0.45,
        "fold_thresholds": {"1": 0.4, "2": 0.5},
        "fold_threshold_defaults": ["2"],
        "plan_warnings": ["Fold 2: the validation rows contain no '1'."],
        "coverage": {
            "test": {"planned": 20, "scored": 18},
            "validation": {"planned": 6, "scored": 6},
            "warnings": ["18 of 20 test rows were scored; the test metrics cover the scored rows only."],
        },
        "exported_model": "fold1_seed42",
        "exported_model_threshold": 0.41,
        "exported_model_threshold_source": "validation",
        "exported_model_test_metrics": {"accuracy": 0.8, "roc_auc": 0.9},
        "members": [
            {"name": "fold1_seed42", "fold": 1, "seed": 42, "status": "ok",
             "predictions_unmatched": [], "validation": {}, "test": {}},
            {"name": "fold2_seed42", "fold": 2, "seed": 42, "status": "ok",
             "predictions_unmatched": ["test"], "validation": {}, "test": {}},
        ],
    }
    html = html_structure.format_ensemble_summary_html(summary, "binary")
    assert "Check before relying on these numbers" in html
    assert "Fold 2: the validation rows contain no &#x27;1&#x27;." in html
    assert "18 of 20 test rows were scored" in html
    assert "fold 1: 0.400; fold 2: 0.500 (default 0.5: could not be tuned)" in html
    assert "Test rows scored</td><td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>18 of 20" in html
    assert "and it trained on one fold&#x27;s training rows" in html
    assert "Its own test metrics (its own fold&#x27;s test rows only):" in html
    assert "at its own validation-tuned threshold 0.410" in html
    summary["exported_model_threshold_source"] = "manual"
    assert "at the manual threshold 0.410" in html_structure.format_ensemble_summary_html(summary, "binary")
    summary["exported_model_threshold_source"] = "default"
    assert "(its validation rows could not tune one)" in html_structure.format_ensemble_summary_html(
        summary, "binary"
    )
    assert "trained (test predictions unmatched)" in html
