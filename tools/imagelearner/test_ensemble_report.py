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
    assert "Mean ± SD (members)" in html
    assert "Ensemble (averaged predictions)" in html
    for name in ("fold1_seed42", "fold1_seed43", "fold2_seed42", "fold2_seed43"):
        assert name in html
    assert "Test Performance Summary" in html
    assert "Decision Threshold (Test)" in html
    assert "Optimized on averaged validation split" in html
    assert "2-fold cross-validation (validation folds) x 2 random seeds" in html
    # binary plots are built from the averaged prediction CSVs
    assert "Threshold plot" in html
    assert "Confusion Matrix" in html


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
