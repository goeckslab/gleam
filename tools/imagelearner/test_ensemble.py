import argparse
import importlib
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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

ensemble = importlib.import_module("ensemble")
constants = importlib.import_module("constants")
utils = importlib.import_module("utils")

SPLIT = constants.SPLIT_COLUMN_NAME
LABEL = constants.LABEL_COLUMN_NAME
IMAGE = constants.IMAGE_PATH_COLUMN_NAME


def _classification_df(n_per_class=12, n_classes=2, groups_of=2, with_test=True):
    rows = []
    for cls in range(n_classes):
        for i in range(n_per_class):
            idx = cls * n_per_class + i
            rows.append(
                {
                    IMAGE: f"images/img_{idx}.png",
                    LABEL: cls,
                    "patient": f"p{idx // groups_of}",
                    SPLIT: 2 if (with_test and i % 4 == 3) else (1 if i % 4 == 2 else 0),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fold planning
# ---------------------------------------------------------------------------

def test_assign_folds_keeps_groups_together_and_covers_every_row():
    df = _classification_df(n_per_class=20)
    folds = ensemble.assign_folds(df, 4, 7, label_column=LABEL, group_column="patient")
    assert len(folds) == len(df)
    assert set(folds) == {0, 1, 2, 3}
    per_group = pd.Series(folds, index=df.index).groupby(df["patient"]).nunique()
    assert (per_group == 1).all()


def test_assign_folds_falls_back_when_stratification_is_impossible():
    df = pd.DataFrame({LABEL: [0] * 3 + [1] * 1, "x": range(4)})
    folds = ensemble.assign_folds(df, 2, 0, label_column=LABEL)
    assert sorted(folds) == [0, 0, 1, 1]


def test_assign_folds_rejects_too_few_rows():
    df = pd.DataFrame({LABEL: [0, 1]})
    with pytest.raises(ValueError):
        ensemble.assign_folds(df, 3, 0, label_column=LABEL)


def test_validation_folds_keep_test_rows_and_rotate_validation():
    df = _classification_df()
    members = ensemble.build_member_plan(
        df,
        cross_validation=True,
        num_folds=3,
        fold_role="validation",
        num_seeds=2,
        random_seed=42,
        validation_size=0.15,
        group_column="patient",
    )
    assert [m.name for m in members] == [
        "fold1_seed42", "fold1_seed43",
        "fold2_seed42", "fold2_seed43",
        "fold3_seed42", "fold3_seed43",
    ]
    base_test = df.index[df[SPLIT] == 2]
    pool = df.index[df[SPLIT] != 2]
    validation_hits = pd.Series(0, index=pool)
    for member in members:
        assert (member.split.loc[base_test] == 2).all()
        assert set(member.split.loc[pool].unique()) <= {0, 1}
        if member.seed == 42:
            validation_hits += (member.split.loc[pool] == 1).astype(int)
    # every pooled row is held out exactly once across the folds of one seed
    assert (validation_hits == 1).all()
    # same fold -> same split regardless of seed
    assert members[0].split.equals(members[1].split)


def test_test_folds_score_every_row_once_and_carve_validation():
    df = _classification_df()
    members = ensemble.build_member_plan(
        df,
        cross_validation=True,
        num_folds=4,
        fold_role="test",
        num_seeds=1,
        random_seed=1,
        validation_size=0.25,
    )
    test_hits = pd.Series(0, index=df.index)
    for member in members:
        assert set(member.split.unique()) == {0, 1, 2}
        test_hits += (member.split == 2).astype(int)
    assert (test_hits == 1).all()


def test_seeds_only_plan_reuses_base_split():
    df = _classification_df()
    members = ensemble.build_member_plan(
        df,
        cross_validation=False,
        num_folds=5,
        fold_role="validation",
        num_seeds=3,
        random_seed=10,
        validation_size=0.15,
    )
    assert [m.seed for m in members] == [10, 11, 12]
    assert all(m.fold is None for m in members)
    assert all(m.split.equals(df[SPLIT].astype(int)) for m in members)


# ---------------------------------------------------------------------------
# Prediction normalization and averaging
# ---------------------------------------------------------------------------

def test_normalize_prediction_frame_maps_boolean_columns_to_numeric_labels():
    df_pred = pd.DataFrame(
        {
            "label_predictions": [True, False],
            "label_probabilities": ["[0.3 0.7]", "[0.8 0.2]"],
            "label_probabilities_False": [0.3, 0.8],
            "label_probabilities_True": [0.7, 0.2],
            "label_probability": [0.7, 0.8],
        }
    )
    out = ensemble.normalize_prediction_frame(df_pred, [0, 1])
    assert ensemble.probability_columns(out.columns) == [
        "label_probabilities_0",
        "label_probabilities_1",
    ]
    assert "label_probabilities" not in out.columns


def test_canonical_label_matches_probability_column_names():
    assert ensemble.canonical_label(1.0) == "1"
    assert ensemble.canonical_label(np.float64(0.0)) == "0"
    assert ensemble.canonical_label(True) == "True"
    assert ensemble.canonical_label(2) == "2"
    assert ensemble.canonical_label(1.5) == "1.5"
    assert ensemble.canonical_label(" cat ") == "cat"


def test_normalize_prediction_frame_handles_float_binary_labels():
    df_pred = pd.DataFrame(
        {"label_probabilities_False": [0.3], "label_probabilities_True": [0.7]}
    )
    out = ensemble.normalize_prediction_frame(df_pred, [0.0, 1.0])
    assert ensemble.probability_columns(out.columns) == [
        "label_probabilities_0",
        "label_probabilities_1",
    ]


def test_apply_binary_threshold_scalar_and_per_row():
    agg = pd.DataFrame(
        {
            "label_probabilities_0": [0.4, 0.35, 0.8],
            "label_probabilities_1": [0.6, 0.65, 0.2],
            "n_models": [2, 2, 2],
        },
        index=[3, 4, 5],
    )
    scalar = ensemble.apply_binary_threshold(agg, 0.7)
    assert scalar["label_predictions"].tolist() == ["0", "0", "0"]
    assert scalar["label_probability"].tolist() == pytest.approx([0.4, 0.35, 0.8])
    assert (scalar[ensemble.DECISION_THRESHOLD_COLUMN] == 0.7).all()
    per_row = ensemble.apply_binary_threshold(agg, [0.5, 0.7, 0.1])
    assert per_row["label_predictions"].tolist() == ["1", "0", "1"]
    assert per_row[ensemble.DECISION_THRESHOLD_COLUMN].tolist() == [0.5, 0.7, 0.1]


def test_classification_statistics_accept_per_row_thresholds_and_float_labels():
    probs = np.array([[0.4, 0.6], [0.3, 0.7]])
    stats = ensemble.compute_classification_statistics(
        ["0", "1"], probs, ["0", "1"], threshold=[0.65, 0.65]
    )
    assert stats["accuracy"] == pytest.approx(1.0)
    float_stats = ensemble.compute_classification_statistics(
        [0.0, 1.0], probs, ["0", "1"], threshold=0.5
    )
    assert float_stats["accuracy"] == pytest.approx(0.5)
    assert float_stats["confusion_matrix"] == [[0, 1], [0, 1]]


def test_normalize_prediction_frame_expands_vector_column():
    df_pred = pd.DataFrame({"label_probabilities": ["[0.2, 0.8]", "[0.9, 0.1]"]})
    out = ensemble.normalize_prediction_frame(df_pred, ["cat", "dog"])
    assert out["label_probabilities_cat"].tolist() == [0.2, 0.9]
    assert out["label_probabilities_dog"].tolist() == [0.8, 0.1]


def test_aggregate_member_predictions_averages_probabilities_by_row():
    a = pd.DataFrame(
        {"label_probabilities_0": [0.2, 0.9], "label_probabilities_1": [0.8, 0.1]},
        index=[5, 7],
    )
    b = pd.DataFrame(
        {"label_probabilities_0": [0.6, 0.5], "label_probabilities_1": [0.4, 0.5]},
        index=[5, 9],
    )
    out = ensemble.aggregate_member_predictions([a, b], "binary")
    assert out.index.tolist() == [5, 7, 9]
    assert out.loc[5, "label_probabilities_0"] == pytest.approx(0.4)
    assert out.loc[5, "label_probabilities_1"] == pytest.approx(0.6)
    assert out.loc[5, "label_predictions"] == "1"
    assert out.loc[7, "label_predictions"] == "0"
    assert out[ensemble.N_MODELS_COLUMN].tolist() == [2, 1, 1]
    assert out.loc[5, "label_probability"] == pytest.approx(0.6)


def test_aggregate_member_predictions_averages_regression_values():
    a = pd.DataFrame({"label_predictions": [1.0, 3.0]}, index=[0, 1])
    b = pd.DataFrame({"label_predictions": [3.0, 5.0]}, index=[0, 1])
    out = ensemble.aggregate_member_predictions([a, b], "regression")
    assert out["label_predictions"].tolist() == [2.0, 4.0]
    assert out[ensemble.N_MODELS_COLUMN].tolist() == [2, 2]


# ---------------------------------------------------------------------------
# Ensemble statistics
# ---------------------------------------------------------------------------

def test_binary_statistics_look_like_ludwig_test_statistics():
    y_true = ["0", "1", "1", "0", "1"]
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])
    stats = ensemble.compute_classification_statistics(y_true, probs, ["0", "1"], threshold=0.5)
    assert utils.detect_output_type({"label": stats}) == "binary"
    assert stats["accuracy"] == pytest.approx(0.8)
    assert stats["roc_auc"] == pytest.approx(5 / 6)
    assert stats["confusion_matrix"] == [[1, 1], [0, 3]]
    assert set(stats["per_class_stats"]) == {"0", "1"}
    assert "roc_curve" in stats and "precision_recall_curve" in stats
    flat = ensemble.flatten_label_statistics(stats)
    assert "confusion_matrix" not in flat and flat["accuracy"] == pytest.approx(0.8)


def test_binary_statistics_apply_decision_threshold():
    y_true = ["0", "1"]
    probs = np.array([[0.4, 0.6], [0.3, 0.7]])
    low = ensemble.compute_classification_statistics(y_true, probs, ["0", "1"], threshold=0.5)
    high = ensemble.compute_classification_statistics(y_true, probs, ["0", "1"], threshold=0.65)
    assert low["accuracy"] == pytest.approx(0.5)
    assert high["accuracy"] == pytest.approx(1.0)


def test_category_statistics_report_balanced_and_micro_accuracy():
    y_true = ["a", "b", "c", "a"]
    probs = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.6, 0.3, 0.1]]
    )
    stats = ensemble.compute_classification_statistics(y_true, probs, ["a", "b", "c"], top_k=2)
    assert utils.detect_output_type({"label": stats}) == "category"
    assert stats["accuracy_micro"] == pytest.approx(0.75)
    assert stats["accuracy"] == pytest.approx((1 + 1 + 0) / 3)
    assert stats["hits_at_k"] == pytest.approx(1.0)
    assert stats["overall_stats"]["avg_f1_score_micro"] == pytest.approx(0.75)
    assert len(stats["confusion_matrix"]) == 3


def test_regression_statistics():
    stats = ensemble.compute_regression_statistics([1.0, 2.0, 4.0], [1.0, 3.0, 2.0])
    assert utils.detect_output_type({"label": stats}) == "regression"
    assert stats["mean_absolute_error"] == pytest.approx(1.0)
    assert stats["root_mean_squared_error"] == pytest.approx(np.sqrt(5 / 3))


def test_summarize_member_metrics_uses_sample_std():
    members = [
        {"test": {"accuracy": 0.8, "roc_auc": 0.9}},
        {"test": {"accuracy": 0.6}},
        {"test": {}},
    ]
    summary = ensemble.summarize_member_metrics(members, "test")
    assert summary["accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["accuracy"]["std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
    assert summary["accuracy"]["n"] == 2
    assert summary["roc_auc"]["std"] == 0.0


# ---------------------------------------------------------------------------
# Workflow orchestration with a fake backend
# ---------------------------------------------------------------------------

class FakeBackend:
    """Writes the files a Ludwig run leaves behind, with seed-dependent probabilities."""

    def __init__(self):
        self.runs = []
        self.report_configs = []

    def prepare_config(self, config_params, split_config):
        config_params["architecture"] = "Resnet 18"
        return "model_type: ecd\n"

    @staticmethod
    def _probabilities(df, seed):
        rng = np.random.RandomState(seed)
        noise = rng.uniform(0.05, 0.3, size=len(df))
        p1 = np.where(df[LABEL].to_numpy() == 1, 1 - noise, noise)
        return pd.DataFrame(
            {
                "label_predictions": p1 >= 0.5,
                "label_probabilities": [f"[{1 - p:.3f} {p:.3f}]" for p in p1],
                "label_probabilities_False": 1 - p1,
                "label_probabilities_True": p1,
                "label_probability": np.maximum(p1, 1 - p1),
            }
        )

    def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
        df = pd.read_csv(dataset_path)
        exp_dir = Path(output_dir) / "experiment_run"
        (exp_dir / "model" / "training_checkpoints").mkdir(parents=True)
        (exp_dir / "model" / "model_weights").write_text("weights")
        (exp_dir / "model" / "training_progress.json").write_text(
            json.dumps({"learning_rate": 0.001, "batch_size": 8, "epoch": 2})
        )
        (exp_dir / "description.json").write_text(
            json.dumps({"config": {"random_seed": random_seed}, "dataset": str(dataset_path)})
        )
        (exp_dir / "predictions.parquet").write_text("parquet")
        test_df = df[df[SPLIT] == 2].reset_index(drop=True)
        self._probabilities(test_df, random_seed).to_csv(exp_dir / "predictions.csv", index=False)
        accuracy = 0.8 + random_seed / 1000.0
        (exp_dir / "test_statistics.json").write_text(
            json.dumps(
                {
                    "label": {
                        "accuracy": accuracy,
                        "roc_auc": 0.9,
                        "per_class_stats": {"False": {}, "True": {}},
                        "confusion_matrix": [[1, 0], [0, 1]],
                    },
                    "combined": {"loss": 0.4},
                }
            )
        )
        (exp_dir / "training_statistics.json").write_text(
            json.dumps(
                {
                    "training": {"label": {"accuracy": [0.6, 0.9], "loss": [0.8, 0.3]}},
                    "validation": {"label": {"accuracy": [0.5, 0.7 + random_seed / 1000.0], "loss": [0.9, 0.5]}},
                }
            )
        )
        self.runs.append((Path(dataset_path).name, random_seed))

    def convert_parquet_to_csv(self, output_dir):
        return None

    def generate_split_predictions(self, output_dir, dataset_path, split_value, output_filename):
        df = pd.read_csv(dataset_path)
        exp_dir = Path(output_dir) / "experiment_run"
        rows = df[df[SPLIT] == split_value].reset_index(drop=True)
        if rows.empty:
            return None
        seed = next(seed for name, seed in reversed(self.runs) if name == Path(dataset_path).name)
        preds = self._probabilities(rows, seed)
        out = pd.concat([rows[[SPLIT, LABEL, IMAGE]], preds], axis=1)
        path = exp_dir / output_filename
        out.to_csv(path, index=False)
        return path

    def generate_plots(self, output_dir):
        return None

    def generate_html_report(self, title, output_dir, config, split_info):
        # The label CSV lives in the workflow's temp dir, which is removed after
        # the run, so snapshot it here as the real report would read it.
        view = pd.read_csv(config["label_column_data_path"])
        self.report_configs.append((dict(config), split_info, view))
        report = Path.cwd() / "image_classification_results_report.html"
        report.write_text("<html>report</html>")
        return report


def _make_dataset(tmp_path, n_per_class=12, label_dtype=None):
    df = _classification_df(n_per_class=n_per_class)
    if label_dtype is not None:
        df[LABEL] = df[LABEL].astype(label_dtype)
    csv_path = tmp_path / "metadata.csv"
    df[[IMAGE, LABEL, "patient", SPLIT]].to_csv(csv_path, index=False)
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in df[IMAGE]:
            zf.writestr(name, b"not really a png")
    return csv_path, zip_path, df


def _make_args(tmp_path, csv_path, zip_path, **overrides):
    values = dict(
        csv_file=csv_path,
        image_zip=zip_path,
        model_name="resnet18",
        use_pretrained=True,
        fine_tune=True,
        epochs=2,
        early_stop=5,
        batch_size=None,
        output_dir=tmp_path / "out",
        validation_size=0.15,
        preprocessing_num_processes=1,
        split_probabilities=[0.7, 0.1, 0.2],
        random_seed=42,
        learning_rate=None,
        augmentation=None,
        image_resize="original",
        threshold=None,
        threshold_mode="auto",
        threshold_metric="f1",
        validation_metric=None,
        target_column=None,
        image_column=None,
        sample_id_column="patient",
        torch_home=None,
        cross_validation=True,
        num_folds=2,
        fold_role="validation",
        num_seeds=2,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_workflow_trains_members_and_averages_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, base_df = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path)
    backend = FakeBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    # 2 folds x 2 seeds, every member trained with its own seed
    assert [seed for _, seed in backend.runs] == [42, 43, 42, 43]
    assert [name for name, _ in backend.runs] == [
        "fold1.csv", "fold1.csv", "fold2.csv", "fold2.csv"
    ]

    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["members_trained"] == 4
    assert summary["task_type"] == "binary"
    assert summary["seeds"] == [42, 43]
    assert summary["exported_model"] == "fold1_seed42"
    assert [m["name"] for m in summary["members"]] == [
        "fold1_seed42", "fold1_seed43", "fold2_seed42", "fold2_seed43"
    ]
    assert summary["members"][1]["validation"]["accuracy"] == pytest.approx(0.743)
    assert summary["member_summary"]["test"]["accuracy"]["n"] == 4
    assert 0 < summary["threshold"] < 1
    assert summary["threshold_source"] == "Optimized on averaged validation split"
    assert summary["threshold_metric"] == "F1"
    assert "fold_thresholds" not in summary
    assert "accuracy" in summary["ensemble_test_metrics"]
    assert "roc_auc" in summary["ensemble_validation_metrics"]

    # layout: member runs kept, first member's model exported, no scratch dirs left
    for name in ("fold1_seed42", "fold2_seed43"):
        member_dir = exp_dir / ensemble.MEMBERS_DIR_NAME / name
        assert (member_dir / "predictions.csv").exists()
        assert (member_dir / "model" / "model_weights").exists()
        assert not (member_dir / "model" / "training_checkpoints").exists()
        assert not (member_dir / "predictions.parquet").exists()
    assert (exp_dir / "model" / "model_weights").exists()
    assert (exp_dir / "description.json").exists()
    assert not (args.output_dir / "ensemble_members").exists()

    # averaged test predictions cover the fixed test rows, scored by all 4 members
    preds = pd.read_csv(exp_dir / "predictions.csv")
    n_test = int((base_df[SPLIT] == 2).sum())
    assert len(preds) == n_test
    assert (preds[SPLIT] == 2).all()
    assert (preds[ensemble.N_MODELS_COLUMN] == 4).all()
    assert list(preds.columns[:3]) == [SPLIT, LABEL, IMAGE]
    assert {"label_probabilities_0", "label_probabilities_1"} <= set(preds.columns)
    member_probs = [
        pd.read_csv(exp_dir / ensemble.MEMBERS_DIR_NAME / name / "predictions.csv")["label_probabilities_True"]
        for name in ("fold1_seed42", "fold1_seed43", "fold2_seed42", "fold2_seed43")
    ]
    expected = sum(member_probs) / 4
    assert preds["label_probabilities_1"].to_numpy() == pytest.approx(expected.to_numpy())
    # exported labels follow the selected threshold, not arg-max
    threshold = summary["threshold"]
    assert np.allclose(preds[ensemble.DECISION_THRESHOLD_COLUMN], threshold)
    expected_labels = np.where(preds["label_probabilities_1"] >= threshold, 1, 0)
    assert preds["label_predictions"].astype(int).tolist() == expected_labels.tolist()

    # out-of-fold validation predictions cover every pooled row exactly once per seed
    val = pd.read_csv(exp_dir / "validation_predictions.csv")
    assert len(val) == int((base_df[SPLIT] != 2).sum())
    assert (val[ensemble.N_MODELS_COLUMN] == 2).all()

    test_stats = json.loads((exp_dir / "test_statistics.json").read_text())
    assert utils.detect_output_type(test_stats) == "binary"

    # the report sees the ensemble summary and a label CSV whose split column
    # marks out-of-fold rows as validation and the fixed test rows as test
    config, split_info, view = backend.report_configs[-1]
    assert config["ensemble"]["members_planned"] == 4
    assert config["threshold"] == summary["threshold"]
    assert int((view[SPLIT] == 2).sum()) == n_test
    assert int((view[SPLIT] == 1).sum()) == len(val)
    assert "2-fold cross-validation (validation folds) x 2 random seeds" in split_info
    # dataset overview keeps the original split (members train on its train+val rows)
    assert config["label_split_counts"][0]["train"] > 0
    assert config["label_split_counts"][0]["test"] == n_test // 2
    # the report also gets the per-fold layout: fixed test rows, rotating validation
    layout = config["ensemble"]["fold_layout"]
    assert [row["fold"] for row in layout] == [1, 2]
    assert all(row["test"] == n_test for row in layout)
    assert sum(row["validation"] for row in layout) == len(base_df) - n_test
    assert sum(row["test"] for row in config["ensemble"]["label_counts"]) == n_test
    assert sum(row["rows"] for row in config["ensemble"]["label_counts"]) == len(base_df) - n_test


def test_workflow_test_folds_score_every_row_out_of_fold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, base_df = _make_dataset(tmp_path)
    args = _make_args(
        tmp_path, csv_path, zip_path, fold_role="test", num_folds=3, num_seeds=1
    )
    backend = FakeBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    exp_dir = args.output_dir / "experiment_run"
    preds = pd.read_csv(exp_dir / "predictions.csv")
    assert len(preds) == len(base_df)
    assert (preds[ensemble.N_MODELS_COLUMN] == 1).all()
    summary = json.loads((exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["fold_role"] == "test"
    assert summary["rows"]["test"] == len(base_df)
    # one threshold per fold, tuned without that fold's test rows, applied per row
    assert sorted(summary["fold_thresholds"]) == ["1", "2", "3"]
    assert "per test fold" in summary["threshold_source"]
    assert summary["threshold"] == pytest.approx(np.mean(list(summary["fold_thresholds"].values())))
    applied = set(np.round(preds[ensemble.DECISION_THRESHOLD_COLUMN], 6))
    assert applied <= {round(v, 6) for v in summary["fold_thresholds"].values()}
    config, split_info, view = backend.report_configs[-1]
    assert (view[SPLIT] == 2).all()
    assert "original test split is ignored" in split_info
    # dataset overview shows every row as test
    assert all(row["train"] == 0 and row["validation"] == 0 for row in config["label_split_counts"])
    # per-fold layout: each row is tested in exactly one fold
    layout = summary["fold_layout"]
    assert [row["fold"] for row in layout] == [1, 2, 3]
    assert sum(row["test"] for row in layout) == len(base_df)
    assert all(row["validation"] > 0 for row in layout)
    assert all(row["test"] == 0 for row in summary["label_counts"])
    assert sum(row["rows"] for row in summary["label_counts"]) == len(base_df)


def test_workflow_seeds_only_reuses_split_and_counts_members(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, base_df = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, cross_validation=False, num_seeds=3)
    backend = FakeBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    assert [seed for _, seed in backend.runs] == [42, 43, 44]
    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["cross_validation"] is False
    assert [m["fold"] for m in summary["members"]] == [None, None, None]
    preds = pd.read_csv(exp_dir / "predictions.csv")
    assert (preds[ensemble.N_MODELS_COLUMN] == 3).all()
    val = pd.read_csv(exp_dir / "validation_predictions.csv")
    assert len(val) == int((base_df[SPLIT] == 1).sum())


def test_workflow_handles_float_binary_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path, label_dtype=float)
    args = _make_args(tmp_path, csv_path, zip_path, num_seeds=1)
    backend = FakeBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    # the fake scores the true class high, so matching labels to columns gives good metrics
    assert summary["ensemble_test_metrics"]["accuracy"] > 0.9
    assert summary["ensemble_test_metrics"]["roc_auc"] > 0.9
    preds = pd.read_csv(exp_dir / "predictions.csv")
    assert set(preds[LABEL].astype(str)) == {"0", "1"}
    assert {"label_probabilities_0", "label_probabilities_1"} <= set(preds.columns)


def test_workflow_continues_when_a_member_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, num_seeds=1)

    class FlakyBackend(FakeBackend):
        def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
            if Path(dataset_path).name == "fold2.csv" and random_seed == 42:
                raise RuntimeError("CUDA out of memory")
            super().run_experiment(dataset_path, config_path, output_dir, random_seed)

    backend = FlakyBackend()
    image_workflow.ImageLearnerCLI(args, backend).run()

    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["members_trained"] == 1
    assert [m["status"] for m in summary["members"]] == ["ok", "failed"]
    preds = pd.read_csv(exp_dir / "predictions.csv")
    assert (preds[ensemble.N_MODELS_COLUMN] == 1).all()


def test_single_model_path_is_unchanged_without_ensemble_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, cross_validation=False, num_seeds=1)
    backend = FakeBackend()

    image_workflow.ImageLearnerCLI(args, backend).run()

    assert backend.runs == [("processed_data_for_ludwig.csv", 42)]
    exp_dir = args.output_dir / "experiment_run"
    assert not (exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).exists()
    config, _, _ = backend.report_configs[-1]
    assert "ensemble" not in config
