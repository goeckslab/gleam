import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


def _install_import_stubs():
    sys.modules.setdefault("shap", types.ModuleType("shap"))

    pycaret = sys.modules.setdefault("pycaret", types.ModuleType("pycaret"))
    classification = sys.modules.setdefault(
        "pycaret.classification", types.ModuleType("pycaret.classification")
    )
    regression = sys.modules.setdefault(
        "pycaret.regression", types.ModuleType("pycaret.regression")
    )
    utils = sys.modules.setdefault("pycaret.utils", types.ModuleType("pycaret.utils"))
    generic = sys.modules.setdefault(
        "pycaret.utils.generic", types.ModuleType("pycaret.utils.generic")
    )

    class ClassificationExperiment:
        pass

    class RegressionExperiment:
        pass

    def get_label_encoder(*args, **kwargs):
        return None

    classification.ClassificationExperiment = ClassificationExperiment
    regression.RegressionExperiment = RegressionExperiment
    generic.get_label_encoder = get_label_encoder
    pycaret.classification = classification
    pycaret.regression = regression
    pycaret.utils = utils
    utils.generic = generic


_install_import_stubs()

from base_model_trainer import BaseModelTrainer, _add_pr_auc_metric_if_supported
from pycaret_classification import (
    ClassificationModelTrainer,
    _should_skip_pycaret_plot,
)
from pycaret_predict import _add_pr_auc_metric_if_supported as add_predict_pr_auc


class FakeModel:
    classes_ = [1, 2, 3, 4, 5]

    def predict(self, X=None):
        n_rows = len(X) if X is not None else 0
        return np.resize(np.asarray(self.classes_), n_rows)

    def predict_proba(self, X=None):
        n_rows = len(X) if X is not None else 0
        proba = np.full((n_rows, len(self.classes_)), 0.05)
        if n_rows:
            proba[np.arange(n_rows), np.arange(n_rows) % len(self.classes_)] = 0.8
        return proba


class FakeExperiment:
    def __init__(self, is_multiclass, always_shape_error=False):
        self.is_multiclass = is_multiclass
        self.always_shape_error = always_shape_error
        self.metric_added = False
        self.compare_kwargs = None
        self.predict_called = False
        self.predict_kwargs = None
        self.X_test = pd.DataFrame({"feature": range(10)})
        self.y_test = pd.Series([1, 2, 3, 4, 5] * 2)

    def add_metric(self, **kwargs):
        self.metric_added = True
        self.metric_kwargs = kwargs

    def compare_models(self, **kwargs):
        self.compare_kwargs = kwargs
        return FakeModel()

    def pull(self):
        return pd.DataFrame({"AUC": [0.75]})

    def predict_model(self, model, **kwargs):
        self.predict_called = True
        self.predict_kwargs = kwargs
        if self.always_shape_error or (self.metric_added and self.is_multiclass):
            raise ValueError("Shape of passed values is (60, 5), indices imply (60, 1)")
        return pd.DataFrame()

    def get_config(self, key):
        values = {
            "X_test": self.X_test,
            "X_test_transformed": self.X_test,
            "y_test": self.y_test,
            "y_test_transformed": self.y_test,
        }
        return values.get(key)


class FakePlotExperiment:
    is_multiclass = True

    def __init__(self):
        self.plot_calls = []

    def plot_model(self, model, plot, save=True, plot_kwargs=None):
        self.plot_calls.append(plot)
        if plot == "threshold":
            raise AssertionError("threshold plot should be skipped for multiclass")
        return f"{plot}.png"


def test_training_pr_auc_metric_is_skipped_for_multiclass():
    exp = FakeExperiment(is_multiclass=True)

    added = _add_pr_auc_metric_if_supported(exp)

    assert added is False
    assert exp.metric_added is False


def test_training_pr_auc_metric_is_added_for_binary():
    exp = FakeExperiment(is_multiclass=False)

    added = _add_pr_auc_metric_if_supported(exp)

    assert added is True
    assert exp.metric_added is True
    assert exp.metric_kwargs["target"] == "pred_proba"


def test_multiclass_train_model_does_not_register_pr_auc_before_predict_model(tmp_path):
    trainer = BaseModelTrainer(
        input_file="unused.tsv",
        target_col="1",
        output_dir=str(tmp_path),
        task_type="classification",
        random_seed=42,
        best_model_metric="PR-AUC",
    )
    trainer.exp = FakeExperiment(is_multiclass=True)

    trainer.train_model()

    assert trainer.exp.metric_added is False
    assert trainer.exp.compare_kwargs["sort"] == "AUC"
    assert trainer.exp.predict_called is True
    assert trainer.exp.predict_kwargs == {}


def test_multiclass_shape_error_falls_back_to_gleam_test_metrics(tmp_path):
    trainer = BaseModelTrainer(
        input_file="unused.tsv",
        target_col="1",
        output_dir=str(tmp_path),
        task_type="classification",
        random_seed=42,
        best_model_metric="AUC",
        probability_threshold=0.7,
    )
    trainer.exp = FakeExperiment(is_multiclass=True, always_shape_error=True)

    trainer.train_model()

    assert trainer.exp.predict_called is True
    assert trainer.exp.predict_kwargs == {}
    assert "Accuracy" in trainer.test_result_df.columns
    assert "ROC-AUC" in trainer.test_result_df.columns
    assert "PR-AUC" in trainer.test_result_df.columns
    assert len(trainer.test_result_df) == 1


def test_predict_evaluator_pr_auc_metric_is_skipped_for_multiclass():
    exp = FakeExperiment(is_multiclass=True)

    added = add_predict_pr_auc(exp)

    assert added is False
    assert exp.metric_added is False


def test_multiclass_threshold_plot_is_skipped():
    exp = FakePlotExperiment()
    trainer = object.__new__(ClassificationModelTrainer)
    trainer.best_model = FakeModel()
    trainer.exp = exp
    trainer.plots = {}

    trainer.generate_plots()

    assert _should_skip_pycaret_plot("threshold", exp) is True
    assert "threshold" not in exp.plot_calls
    assert "threshold" not in trainer.plots
    assert "auc" in exp.plot_calls
