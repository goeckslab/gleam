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


def test_assign_folds_rejects_fewer_sample_ids_than_folds():
    df = pd.DataFrame({LABEL: [0, 1] * 4, "patient": ["a", "b"] * 4})
    with pytest.raises(ensemble.EnsembleSetupError, match="Use at most 2 folds"):
        ensemble.assign_folds(df, 3, 0, group_column="patient")


def test_grouped_fallback_folds_follow_the_seed_and_keep_groups_together():
    df = pd.DataFrame({"patient": [f"p{i // 2}" for i in range(40)]})
    by_seed = {
        seed: ensemble.assign_folds(df, 4, seed, group_column="patient") for seed in (1, 2)
    }
    for folds in by_seed.values():
        assert (pd.Series(folds).groupby(df["patient"]).nunique() == 1).all()
        assert sorted(np.bincount(folds)) == [10, 10, 10, 10]
    assert not np.array_equal(by_seed[1], by_seed[2])


def test_member_plan_refuses_classes_with_fewer_rows_than_folds():
    df = _classification_df(n_per_class=12)
    df.loc[df[LABEL] == 1, LABEL] = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    with pytest.raises(ensemble.EnsembleSetupError, match="than the 3 folds.*Use at most 2 folds"):
        ensemble.build_member_plan(
            df, cross_validation=True, num_folds=3, fold_role="validation",
            num_seeds=1, random_seed=0, validation_size=0.15,
        )


def test_member_plan_advice_for_a_single_row_class():
    df = _classification_df(n_per_class=12)
    df.loc[0, LABEL] = 5
    with pytest.raises(ensemble.EnsembleSetupError, match="Add rows for these classes or remove them"):
        ensemble.build_member_plan(
            df, cross_validation=True, num_folds=2, fold_role="test",
            num_seeds=1, random_seed=0, validation_size=0.15,
        )


def test_test_folds_refuse_a_user_supplied_test_split():
    df = _classification_df()
    kwargs = dict(cross_validation=True, num_folds=2, fold_role="test", num_seeds=1, random_seed=0,
                  validation_size=0.15)
    with pytest.raises(ensemble.EnsembleSetupError, match="Use validation folds"):
        ensemble.build_member_plan(df, user_test_split=True, **kwargs)
    # a test split the tool generated itself may be folded back
    assert len(ensemble.build_member_plan(df, user_test_split=False, **kwargs)) == 2


def test_plan_warnings_name_folds_whose_validation_misses_a_class():
    df = pd.DataFrame({LABEL: [0, 0, 1, 1, 0, 1]})
    split_a = pd.Series([0, 1, 0, 1, 2, 2])
    split_b = pd.Series([1, 0, 0, 0, 2, 2])
    members = [
        ensemble.MemberSpec("fold1_seed1", 1, 1, split_a),
        ensemble.MemberSpec("fold1_seed2", 2, 1, split_a),
        ensemble.MemberSpec("fold2_seed1", 1, 2, split_b),
    ]
    assert ensemble.plan_warnings(df, members) == ["Fold 2: the validation rows contain no '1'."]


def test_plan_warnings_name_folds_that_train_without_a_class():
    # every 'c' row shares one sample ID, so one fold holds the whole class out
    df = pd.DataFrame(
        {
            LABEL: ["a"] * 20 + ["b"] * 20 + ["c"] * 5,
            "patient": list(range(40)) + [99] * 5,
            SPLIT: 0,
        }
    )
    for fold_role in ensemble.FOLD_ROLES:
        members = ensemble.build_member_plan(
            df, cross_validation=True, num_folds=5, fold_role=fold_role, num_seeds=1,
            random_seed=42, validation_size=0.15, group_column="patient",
        )
        untrained = [m.fold for m in members if "c" not in set(df[LABEL][m.split == 0])]
        assert len(untrained) == 1
        warnings = ensemble.plan_warnings(df, members)
        expected = f"Fold {untrained[0]}: the training rows contain no 'c', so this fold's models cannot predict it."
        assert expected in warnings
        assert sum("training rows" in message for message in warnings) == 1


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


@pytest.mark.parametrize(
    "labels, false_label, true_label",
    [([0, 2], "0", "2"), ([0, -1], "0", "-1"), ([2.5, 0.0], "0", "2.5")],
)
def test_normalize_prediction_frame_maps_boolean_columns_zero_nonzero(labels, false_label, true_label):
    # Ludwig casts numeric binary targets to bool: zero -> False, nonzero -> True.
    df_pred = pd.DataFrame(
        {"label_probabilities_False": [0.3, 0.8], "label_probabilities_True": [0.7, 0.2]}
    )
    out = ensemble.normalize_prediction_frame(df_pred, labels)
    assert out[f"label_probabilities_{false_label}"].tolist() == [0.3, 0.8]
    assert out[f"label_probabilities_{true_label}"].tolist() == [0.7, 0.2]
    assert "label_probabilities_False" not in out.columns


def test_boolean_probability_labels_rejects_unmappable_labels():
    assert ensemble.boolean_probability_labels([1, 2]) is None
    assert ensemble.boolean_probability_labels([0, 0.0]) is None
    assert ensemble.boolean_probability_labels(["cat", "dog"]) is None
    assert ensemble.boolean_probability_labels([True, False]) is None
    assert ensemble.boolean_probability_labels([0, np.nan, 3]) == {"False": "0", "True": "3"}


def test_ensemble_statistics_for_zero_two_binary_labels():
    members = [
        ensemble.normalize_prediction_frame(
            pd.DataFrame(
                {
                    "label_probabilities_False": false_probs,
                    "label_probabilities_True": [1 - p for p in false_probs],
                }
            ),
            [0, 2],
        )
        for false_probs in ([0.9, 0.2, 0.6, 0.3], [0.7, 0.4, 0.8, 0.1])
    ]
    agg = ensemble.aggregate_member_predictions(members, "binary")
    names = [ensemble.class_name_from_column(c) for c in ensemble.probability_columns(agg.columns)]
    assert sorted(names) == ["0", "2"]
    probs = agg[[f"label_probabilities_{name}" for name in names]].to_numpy(dtype=float)
    stats = ensemble.compute_classification_statistics([0, 2, 0, 2], probs, names, threshold=0.5)
    assert stats["accuracy"] == pytest.approx(1.0)
    assert ensemble.positive_class_index(names) == names.index("2")


@pytest.mark.parametrize(
    "labels, expected",
    [(["0", "2"], "2"), (["-2", "0"], "-2"), (["0.0", "2.5"], "2.5"), (["1", "2"], None),
     (["0", "0.0"], None), (["False", "True"], None), (["0", "nan"], None), (["0", "1", "2"], None)],
)
def test_nonzero_binary_label(labels, expected):
    assert utils.nonzero_binary_label(labels) == expected


@pytest.mark.parametrize(
    "names, positive",
    [(["0", "-2"], "-2"), (["-2", "0"], "-2"), (["0", "2"], "2"), (["0", "1"], "1"),
     (["False", "True"], "True"), (["cat", "dog"], "dog")],
)
def test_positive_class_index_keeps_ludwig_nonzero_class(names, positive):
    assert names[ensemble.positive_class_index(names)] == positive


def test_apply_binary_threshold_thresholds_nonzero_class_like_ludwig():
    # Ludwig thresholds P(True) = P(nonzero label): P(-2) = 0.7 below 0.8 predicts 0.
    agg = pd.DataFrame(
        {"label_probabilities_0": [0.3], "label_probabilities_-2": [0.7], "n_models": [2]},
        index=[0],
    )
    assert ensemble.apply_binary_threshold(agg, 0.8)["label_predictions"].tolist() == ["0"]
    assert ensemble.apply_binary_threshold(agg, 0.5)["label_predictions"].tolist() == ["-2"]


def test_normalize_prediction_frame_maps_ludwig_float_and_spaced_class_names():
    # Ludwig names category classes with str(value): a float label 1.0 becomes "1.0".
    df_pred = pd.DataFrame(
        {
            "label_probabilities_1.0": [0.7, 0.1],
            "label_probabilities_2.0": [0.2, 0.8],
            "label_probabilities_3.5": [0.1, 0.1],
        }
    )
    out = ensemble.normalize_prediction_frame(df_pred, [1.0, 2.0, 3.5])
    assert ensemble.probability_columns(out.columns) == [
        "label_probabilities_1",
        "label_probabilities_2",
        "label_probabilities_3.5",
    ]
    assert out["label_probabilities_2"].tolist() == [0.2, 0.8]
    spaced = ensemble.normalize_prediction_frame(
        pd.DataFrame({"label_probabilities_cat": [0.6], "label_probabilities_dog": [0.4]}), [" cat ", "dog "]
    )
    assert ensemble.probability_columns(spaced.columns) == ["label_probabilities_cat", "label_probabilities_dog"]


def test_normalize_prediction_frame_expands_vector_in_ludwig_class_order():
    df_pred = pd.DataFrame({"label_probabilities": ["[0.6, 0.3, 0.1]"]})
    # idx2str is frequency-ordered, not alphabetical.
    out = ensemble.normalize_prediction_frame(df_pred, ["a", "b", "c"], class_order=["b", "c", "a"])
    assert out["label_probabilities_b"].tolist() == [0.6]
    assert out["label_probabilities_c"].tolist() == [0.3]
    assert out["label_probabilities_a"].tolist() == [0.1]


def test_canonical_class_names_keeps_ambiguous_names():
    assert ensemble.canonical_class_names(["1.0", "2.0"], [1, 2]) == ["1", "2"]
    assert ensemble.canonical_class_names(["1", "1.0"], [1]) == ["1", "1.0"]
    assert ensemble.canonical_class_names(["False", "True"], [1, 2]) == ["False", "True"]


def test_classification_statistics_refuse_labels_that_match_no_class():
    # Labels 1/2: Ludwig's False/True columns cannot be mapped back, so fail loudly.
    with pytest.raises(ValueError, match="matches a predicted class"):
        ensemble.compute_classification_statistics(
            [1, 2, 1], np.array([[0.4, 0.6], [0.3, 0.7], [0.5, 0.5]]), ["False", "True"]
        )


def test_regression_percentage_errors_match_ludwig_definitions():
    stats = ensemble.compute_regression_statistics([0.0, 2.0], [0.5, 1.0])
    # torchmetrics MAPE clamps |y| at 1.17e-06; Ludwig's RMSPE adds 1e-4 to zero targets
    assert stats["mean_absolute_percentage_error"] == pytest.approx((0.5 / 1.17e-06 + 0.5) / 2)
    assert stats["root_mean_squared_percentage_error"] == pytest.approx(
        np.sqrt(((0.5 / 1e-4) ** 2 + 0.5 ** 2) / 2)
    )


def test_argument_checker_caps_the_number_of_members(tmp_path):
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("image_path,label\n")
    args = argparse.Namespace(
        validation_size=0.15, csv_file=csv_path, image_zip=tmp_path, threshold=None,
        cross_validation=True, num_folds=10, num_seeds=6, augmentation=None,
    )
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        utils.argument_checker(args, parser)
    args.num_seeds = 5
    utils.argument_checker(args, parser)


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

def test_statistics_leave_out_rows_whose_label_matches_no_class():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.9, 0.1]])
    known = ["0", "1", "1", "0"]
    # the two extra rows would count as a false positive and a true negative if kept as negatives
    with_unknown = ensemble.compute_classification_statistics(
        known + [float("nan"), "other"], probs, ["0", "1"], threshold=[0.5, 0.5, 0.5, 0.5, 0.2, 0.2]
    )
    only_known = ensemble.compute_classification_statistics(known, probs[:4], ["0", "1"], threshold=0.5)
    assert with_unknown == only_known
    assert int(np.sum(with_unknown["confusion_matrix"])) == 4
    assert with_unknown["accuracy"] == 1.0

    multi = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    with_unknown = ensemble.compute_classification_statistics(["a", "b", "c", "zzz"], multi, ["a", "b", "c"])
    assert with_unknown == ensemble.compute_classification_statistics(["a", "b", "c"], multi[:3], ["a", "b", "c"])


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


def _make_dataset(tmp_path, n_per_class=12, label_dtype=None, with_split=True):
    df = _classification_df(n_per_class=n_per_class)
    if label_dtype is not None:
        df[LABEL] = df[LABEL].astype(label_dtype)
    csv_path = tmp_path / "metadata.csv"
    columns = [IMAGE, LABEL, "patient"] + ([SPLIT] if with_split else [])
    df[columns].to_csv(csv_path, index=False)
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
    # Ludwig's own (last-epoch, 0.5) numbers are kept apart from the like-for-like re-scoring.
    assert summary["members"][1]["ludwig"]["validation"]["accuracy"] == pytest.approx(0.743)
    assert summary["members"][1]["validation"]["accuracy"] == pytest.approx(1.0)
    assert summary["members"][1]["predictions_unmatched"] == []
    assert summary["member_summary"]["test"]["accuracy"]["n"] == 4
    assert summary["coverage"]["test"] == {"planned": int((base_df[SPLIT] == 2).sum()),
                                           "scored": int((base_df[SPLIT] == 2).sum())}
    assert summary["coverage"]["warnings"] == []
    assert 0 < summary["exported_model_threshold"] < 1
    assert "accuracy" in summary["exported_model_test_metrics"]
    assert 0 < summary["threshold"] < 1
    assert summary["threshold_source"].startswith("Optimized on out-of-fold validation predictions")
    assert summary["threshold_metric"] == "F1"
    assert "fold_thresholds" not in summary
    assert "accuracy" in summary["ensemble_test_metrics"]
    assert "roc_auc" in summary["ensemble_validation_metrics"]

    # layout: member runs kept, only the exported member keeps weights, no scratch dirs left
    for name in ("fold1_seed42", "fold2_seed43"):
        member_dir = exp_dir / ensemble.MEMBERS_DIR_NAME / name
        assert (member_dir / "predictions.csv").exists()
        assert (member_dir / "model" / "model_weights").exists() == (name == "fold1_seed42")
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
    # No split column: the generated test split may be folded back into cross-validation.
    csv_path, zip_path, base_df = _make_dataset(tmp_path, with_split=False)
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
    assert "the generated test split is folded back in" in split_info
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
    args = _make_args(tmp_path, csv_path, zip_path, num_seeds=2)

    class FlakyBackend(FakeBackend):
        def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
            if Path(dataset_path).name == "fold2.csv" and random_seed == 42:
                raise RuntimeError("CUDA out of memory")
            super().run_experiment(dataset_path, config_path, output_dir, random_seed)

    backend = FlakyBackend()
    image_workflow.ImageLearnerCLI(args, backend).run()

    exp_dir = args.output_dir / "experiment_run"
    summary = json.loads((exp_dir / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["members_trained"] == 3
    assert [m["status"] for m in summary["members"]] == ["ok", "ok", "failed", "ok"]
    preds = pd.read_csv(exp_dir / "predictions.csv")
    assert (preds[ensemble.N_MODELS_COLUMN] == 3).all()
    # every fold still has a trained member, so every validation row is scored
    assert summary["coverage"]["validation"]["scored"] == summary["coverage"]["validation"]["planned"]


def test_workflow_refuses_a_cross_validation_with_an_untrained_fold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, num_seeds=1)

    class FoldFailureBackend(FakeBackend):
        def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
            if Path(dataset_path).name == "fold2.csv":
                raise RuntimeError("CUDA out of memory")
            super().run_experiment(dataset_path, config_path, output_dir, random_seed)

    backend = FoldFailureBackend()
    # fold 2 held out rows nobody scored: no partial ensemble, the job fails with the reason
    with pytest.raises(ensemble.EnsembleRunError, match=r"No member of fold\(s\) 2 trained"):
        image_workflow.ImageLearnerCLI(args, backend).run()

    assert not (args.output_dir / "experiment_run" / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).exists()
    # the member that did train is not left behind in the scratch directory
    assert not (args.output_dir / ensemble.MEMBERS_WORK_DIR_NAME).exists()
    # no stub model or fallback report makes the failed run look like a success
    assert backend.report_configs == []
    assert not (args.output_dir / "experiment_run" / "model").exists()


def test_workflow_fails_when_every_member_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, cross_validation=False, num_seeds=2)

    class BrokenBackend(FakeBackend):
        def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
            raise RuntimeError("CUDA out of memory")

    backend = BrokenBackend()
    with pytest.raises(ensemble.EnsembleRunError, match="All 2 ensemble members failed"):
        image_workflow.ImageLearnerCLI(args, backend).run()
    assert backend.report_configs == []
    assert not (args.output_dir / ensemble.MEMBERS_WORK_DIR_NAME).exists()


def test_workflow_fails_when_no_member_predictions_can_be_matched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, num_seeds=1)

    class MisalignedBackend(FakeBackend):
        def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
            super().run_experiment(dataset_path, config_path, output_dir, random_seed)
            preds = Path(output_dir) / "experiment_run" / "predictions.csv"
            pd.read_csv(preds).iloc[:1].to_csv(preds, index=False)

        def generate_split_predictions(self, output_dir, dataset_path, split_value, output_filename):
            return None

    backend = MisalignedBackend()
    with pytest.raises(ensemble.EnsembleRunError, match="nothing to evaluate"):
        image_workflow.ImageLearnerCLI(args, backend).run()

    assert not (args.output_dir / "experiment_run" / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).exists()
    assert backend.report_configs == []
    assert not (args.output_dir / ensemble.MEMBERS_WORK_DIR_NAME).exists()


@pytest.mark.parametrize("ensemble_options", [{"num_seeds": 1}, {"cross_validation": False, "num_seeds": 1}])
def test_workflow_releases_cached_backend_data_before_the_report(tmp_path, monkeypatch, ensemble_options):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, **ensemble_options)
    events = []

    class TrackingBackend(FakeBackend):
        def release_cached_data(self):
            events.append("release")

        def generate_html_report(self, *args, **kwargs):
            events.append("report")
            return super().generate_html_report(*args, **kwargs)

    image_workflow.ImageLearnerCLI(args, TrackingBackend()).run()

    assert "release" in events
    assert events.index("release") < events.index("report")


@pytest.mark.parametrize("labels, refused", [((1, 2), True), ((-1, 1), True), ((0, 1), False), ((0, 2), False)])
def test_workflow_refuses_numeric_binary_labels_ludwig_cannot_tell_apart(tmp_path, monkeypatch, labels, refused):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, df = _make_dataset(tmp_path)
    df[LABEL] = df[LABEL].map({0: labels[0], 1: labels[1]})
    df[[IMAGE, LABEL, "patient", SPLIT]].to_csv(csv_path, index=False)
    args = _make_args(tmp_path, csv_path, zip_path, cross_validation=False, num_seeds=1)
    cli = image_workflow.ImageLearnerCLI(args, FakeBackend())
    if refused:
        with pytest.raises(ValueError, match="Recode the labels as 0 and 1"):
            cli.run()
    else:
        cli.run()


def test_workflow_fails_before_training_on_bad_ensemble_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)  # the CSV supplies a test split
    args = _make_args(tmp_path, csv_path, zip_path, fold_role="test", num_folds=2, num_seeds=1)
    backend = FakeBackend()
    with pytest.raises(ensemble.EnsembleSetupError):
        image_workflow.ImageLearnerCLI(args, backend).run()
    assert backend.runs == []


def test_workflow_cleans_fold_caches_and_member_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, num_folds=2, num_seeds=2)
    leftovers = []

    class CachingBackend(FakeBackend):
        def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
            dataset_path = Path(dataset_path)
            # what Ludwig leaves next to the CSV
            dataset_path.with_suffix(".training.hdf5").write_text("cache")
            if dataset_path.name == "fold2.csv":
                leftovers.extend(p.name for p in dataset_path.parent.glob("fold1.*"))
            super().run_experiment(dataset_path, config_path, output_dir, random_seed)

        def generate_html_report(self, title, output_dir, config, split_info):
            gradcam = pd.read_csv(config["gradcam_label_path"])
            self.gradcam_test_rows = int((gradcam[SPLIT] == 2).sum())
            return super().generate_html_report(title, output_dir, config, split_info)

    backend = CachingBackend()
    image_workflow.ImageLearnerCLI(args, backend).run()

    assert leftovers == []
    members_dir = args.output_dir / "experiment_run" / ensemble.MEMBERS_DIR_NAME
    kept = sorted(p.parent.parent.name for p in members_dir.glob("*/model/model_weights"))
    assert kept == ["fold1_seed42"]
    # Grad-CAM samples the exported member's own test rows
    assert backend.gradcam_test_rows > 0

    args = _make_args(tmp_path, csv_path, zip_path, num_folds=2, num_seeds=1,
                      output_dir=tmp_path / "keep", keep_member_weights=True)
    image_workflow.ImageLearnerCLI(args, FakeBackend()).run()
    kept = list((tmp_path / "keep" / "experiment_run" / ensemble.MEMBERS_DIR_NAME).glob("*/model/model_weights"))
    assert len(kept) == 2


def test_workflow_without_test_rows_reports_no_unmatched_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, df = _make_dataset(tmp_path)
    df[SPLIT] = df[SPLIT].replace(2, 0)  # train/validation only
    df[[IMAGE, LABEL, "patient", SPLIT]].to_csv(csv_path, index=False)
    args = _make_args(tmp_path, csv_path, zip_path, cross_validation=False, num_seeds=2)

    image_workflow.ImageLearnerCLI(args, FakeBackend()).run()

    summary = json.loads((args.output_dir / "experiment_run" / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert [m["predictions_unmatched"] for m in summary["members"]] == [[], []]
    assert summary["coverage"]["test"] == {"planned": 0, "scored": 0}
    assert summary["coverage"]["warnings"] == []


def test_workflow_rescores_category_members_from_their_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    df = _classification_df(n_per_class=12, n_classes=3)
    csv_path = tmp_path / "metadata.csv"
    df[[IMAGE, LABEL, "patient", SPLIT]].to_csv(csv_path, index=False)
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in df[IMAGE]:
            zf.writestr(name, b"not really a png")
    args = _make_args(tmp_path, csv_path, zip_path, num_folds=2, num_seeds=1)

    class CategoryBackend(FakeBackend):
        @staticmethod
        def _probabilities(frame, seed):
            labels = frame[LABEL].to_numpy()
            # the true class scores highest except for every fourth row, which goes to the next class
            chosen = np.where(np.arange(len(frame)) % 4 == 0, (labels + 1) % 3, labels)
            probs = np.full((len(frame), 3), 0.1)
            probs[np.arange(len(frame)), chosen] = 0.8
            out = pd.DataFrame({f"label_probabilities_{k}": probs[:, k] for k in range(3)})
            out.insert(0, "label_predictions", chosen)
            out["label_probability"] = 0.8
            return out

    image_workflow.ImageLearnerCLI(args, CategoryBackend()).run()

    summary = json.loads((args.output_dir / "experiment_run" / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["task_type"] == "category"
    member = summary["members"][0]
    n_test = int((df[SPLIT] == 2).sum())
    expected = 1 - len(range(0, n_test, 4)) / n_test
    # re-scored from the member's own predictions, not Ludwig's numbers
    assert member["test"]["accuracy_micro"] == pytest.approx(expected)
    assert member["ludwig"]["test"]["accuracy"] == pytest.approx(0.842)
    assert "accuracy_micro" in summary["exported_model_test_metrics"]


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


# ---------------------------------------------------------------------------
# Follow-up review: threshold-tuning rows, seeded group folds, vector fallback,
# test-fold validation size, text binary labels, 0/2 split groups
# ---------------------------------------------------------------------------
def _binary_plan(n_pos):
    """Two test folds over 40 negatives and n_pos positives; validation drawn from training rows."""
    df = pd.DataFrame({LABEL: [0] * 40 + [1] * n_pos, SPLIT: 0})
    return df, ensemble.build_member_plan(
        df, cross_validation=True, num_folds=2, fold_role="test", num_seeds=1,
        random_seed=0, validation_size=0.25,
    )


def test_plan_warnings_flag_thin_threshold_tuning_rows_per_test_fold():
    df, members = _binary_plan(n_pos=8)  # about 1 positive in each fold's validation rows
    warnings = ensemble.plan_warnings(df, members, fold_role="test", tunes_threshold=True)
    thin = [w for w in warnings if "decision threshold rests on very few" in w]
    assert len(thin) == 2
    assert thin[0].startswith("Fold 1: the validation rows hold only ")
    assert "'1' row" in thin[0]
    # a manual threshold is not tuned, so nothing to flag
    assert not any(
        "rests on very few" in w for w in ensemble.plan_warnings(df, members, fold_role="test")
    )


def test_plan_warnings_stay_quiet_when_each_fold_tunes_on_enough_rows():
    df, members = _binary_plan(n_pos=80)  # about 10 positives per fold's validation rows
    warnings = ensemble.plan_warnings(df, members, fold_role="test", tunes_threshold=True)
    assert not any("rests on very few" in w for w in warnings)


def test_plan_warnings_count_pooled_validation_rows_for_one_threshold():
    df = pd.DataFrame({LABEL: [0] * 30 + [1] * 3, SPLIT: [0] * 20 + [1] * 10 + [0, 1, 1]})
    seeds_only = ensemble.build_member_plan(
        df, cross_validation=False, num_folds=2, fold_role="validation", num_seeds=2,
        random_seed=0, validation_size=0.15,
    )
    assert ensemble.plan_warnings(df, seeds_only, tunes_threshold=True) == [
        "The validation rows the decision threshold is tuned on hold only 2 '1' rows; "
        "the threshold rests on very few examples."
    ]
    # validation folds pool every non-test row; 3 positives are still too few
    folds = ensemble.build_member_plan(
        df, cross_validation=True, num_folds=3, fold_role="validation", num_seeds=1,
        random_seed=0, validation_size=0.15,
    )
    pooled = ensemble.plan_warnings(df, folds, fold_role="validation", tunes_threshold=True)
    assert "hold only 3 '1' rows" in pooled[0]


def test_grouped_fallback_folds_follow_the_seed_when_every_group_size_differs():
    # 12 sample IDs with 1..12 images each: sorting by size alone would ignore the seed
    patients = [f"p{size}" for size in range(1, 13) for _ in range(size)]
    df = pd.DataFrame({"patient": patients})
    by_seed = [ensemble.assign_folds(df, 3, seed, group_column="patient") for seed in range(5)]
    for folds in by_seed:
        assert (pd.Series(folds).groupby(df["patient"]).nunique() == 1).all()
        assert set(folds) == {0, 1, 2}
        # the 78 rows stay roughly balanced (26 per fold), within the largest group's size
        assert max(np.bincount(folds)) - min(np.bincount(folds)) <= 12
    assert len({tuple(folds) for folds in by_seed}) > 1


@pytest.mark.parametrize(
    "labels, vector, expected",
    [
        # numeric: [P(False), P(True)] = [P(0), P(-2)], whatever the sorted order of the labels
        ([0, -2], "[0.7, 0.3]", {"0": 0.7, "-2": 0.3}),
        ([2, 0], "[0.7, 0.3]", {"0": 0.7, "2": 0.3}),
    ],
)
def test_normalize_prediction_frame_names_a_numeric_binary_vector_by_boolean(labels, vector, expected):
    out = ensemble.normalize_prediction_frame(pd.DataFrame({"label_probabilities": [vector]}), labels)
    assert {ensemble.class_name_from_column(c): out[c].iloc[0] for c in out.columns if c != "label_predictions"} == (
        expected
    )


def test_normalize_prediction_frame_names_a_text_binary_vector_by_bool2str():
    # Ludwig makes the alphabetically first label True: bool2str = [False label, True label]
    out = ensemble.normalize_prediction_frame(
        pd.DataFrame({"label_probabilities": ["[0.9, 0.1]"]}), ["case", "control"], class_order=["control", "case"]
    )
    assert out["label_probabilities_control"].tolist() == [0.9]
    assert out["label_probabilities_case"].tolist() == [0.1]


def test_member_outputs_read_bool2str_for_a_text_binary_vector(tmp_path):
    image_workflow = importlib.import_module("image_workflow")
    exp_dir = tmp_path / "experiment_run"
    (exp_dir / "model").mkdir(parents=True)
    (exp_dir / "model" / "training_set_metadata.json").write_text(
        json.dumps({LABEL: {"bool2str": ["control", "case"], "fallback_true_label": "case"}})
    )
    cli = image_workflow.ImageLearnerCLI(argparse.Namespace(), FakeBackend())
    assert cli._ludwig_label_classes(exp_dir) is None  # confusion-matrix order is unchanged
    assert cli._ludwig_label_classes(exp_dir, include_binary=True) == ["control", "case"]


@pytest.mark.parametrize(
    "labels, same",
    [
        (("yes", "true"), True),
        (("No", "FALSE"), True),
        (("maybe", "yes"), True),  # "maybe" sorts first, so it is the fallback True label
        (("no", "other"), True),  # "no" sorts first but is False; "other" is not the fallback
        (("yes", "no"), False),
        (("benign", "malignant"), False),
        (("case", "control"), False),
    ],
)
def test_ludwig_text_binary_truth_follows_ludwig(labels, same):
    truth = utils.ludwig_text_binary_truth(labels)
    assert (truth[labels[0]] == truth[labels[1]]) is same
    if labels == ("case", "control"):
        assert truth == {"case": True, "control": False}


@pytest.mark.parametrize("labels, refused", [(("yes", "true"), True), (("maybe", "yes"), True), (("benign", "malignant"), False)])
def test_workflow_refuses_text_binary_labels_ludwig_cannot_tell_apart(tmp_path, monkeypatch, labels, refused):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, df = _make_dataset(tmp_path)
    df[LABEL] = df[LABEL].map({0: labels[0], 1: labels[1]})
    df[[IMAGE, LABEL, "patient", SPLIT]].to_csv(csv_path, index=False)
    args = _make_args(tmp_path, csv_path, zip_path, cross_validation=False, num_seeds=1)
    cli = image_workflow.ImageLearnerCLI(args, FakeBackend())
    if refused:
        with pytest.raises(ValueError, match="reads as the same boolean"):
            cli.run()
    else:
        cli.run()


@pytest.mark.parametrize("with_split, expected", [(False, 0.1 / 0.8), (True, 0.15)])
def test_test_folds_draw_validation_by_the_form_proportions(tmp_path, monkeypatch, with_split, expected):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, df = _make_dataset(tmp_path, n_per_class=40, with_split=with_split)
    if with_split:
        df[SPLIT] = df[SPLIT].replace(2, 0)  # a user split without test rows
        df[[IMAGE, LABEL, "patient", SPLIT]].to_csv(csv_path, index=False)
    # without sample IDs the validation rows are not rounded to whole groups
    args = _make_args(tmp_path, csv_path, zip_path, fold_role="test", num_folds=2, num_seeds=1,
                      sample_id_column=None)
    backend = FakeBackend()
    image_workflow.ImageLearnerCLI(args, backend).run()

    summary = json.loads((args.output_dir / "experiment_run" / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    assert summary["fold_validation_size"] == pytest.approx(expected)
    for row in summary["fold_layout"]:
        share = row["validation"] / (row["train"] + row["validation"])
        assert share == pytest.approx(expected, abs=0.03)
    _, split_info, _ = backend.report_configs[-1]
    assert f"each fold draws {expected * 100:.1f}% of its training rows as validation rows" in split_info


def test_validation_fold_summary_records_score_spread(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_workflow = importlib.import_module("image_workflow")
    csv_path, zip_path, _ = _make_dataset(tmp_path)
    args = _make_args(tmp_path, csv_path, zip_path, num_seeds=1)
    image_workflow.ImageLearnerCLI(args, FakeBackend()).run()
    summary = json.loads((args.output_dir / "experiment_run" / ensemble.ENSEMBLE_SUMMARY_FILE_NAME).read_text())
    spread = summary["score_spread"]
    assert set(spread) == {"validation", "test"}
    assert all(isinstance(value, float) and value >= 0 for value in spread.values())


def test_split_0_2_keeps_sample_ids_out_of_both_training_and_validation():
    split_data = importlib.import_module("split_data")
    df = pd.DataFrame(
        {
            LABEL: [i % 2 for i in range(120)],
            "patient": [f"p{i // 4}" for i in range(120)],
            SPLIT: [2 if (i // 4) % 5 == 0 else 0 for i in range(120)],
        }
    )
    out = split_data.split_data_0_2(df, SPLIT, validation_size=0.2, random_state=3, label_column=LABEL,
                                    group_column="patient")
    assert (out.loc[df[SPLIT] == 2, SPLIT] == 2).all()  # test rows untouched
    assert set(out[SPLIT]) == {0, 1, 2}
    splits_per_patient = out.groupby("patient")[SPLIT].nunique()
    assert (splits_per_patient == 1).all()
    validation_share = (out[SPLIT] == 1).sum() / (df[SPLIT] == 0).sum()
    assert validation_share == pytest.approx(0.2, abs=0.05)
