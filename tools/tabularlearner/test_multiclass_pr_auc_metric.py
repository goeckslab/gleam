import sys
import types
from pathlib import Path

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

    class ClassificationExperiment:
        pass

    class RegressionExperiment:
        pass

    classification.ClassificationExperiment = ClassificationExperiment
    regression.RegressionExperiment = RegressionExperiment
    pycaret.classification = classification
    pycaret.regression = regression


_install_import_stubs()

from base_model_trainer import BaseModelTrainer, _add_pr_auc_metric_if_supported
from pycaret_predict import _add_pr_auc_metric_if_supported as add_predict_pr_auc


class FakeModel:
    def predict(self, X=None):
        return []


class FakeExperiment:
    def __init__(self, is_multiclass):
        self.is_multiclass = is_multiclass
        self.metric_added = False
        self.compare_kwargs = None
        self.predict_called = False

    def add_metric(self, **kwargs):
        self.metric_added = True
        self.metric_kwargs = kwargs

    def compare_models(self, **kwargs):
        self.compare_kwargs = kwargs
        return FakeModel()

    def pull(self):
        return pd.DataFrame({"AUC": [0.75]})

    def predict_model(self, model, **kwargs):
        if self.metric_added and self.is_multiclass:
            raise ValueError("Shape of passed values is (60, 5), indices imply (60, 1)")
        self.predict_called = True
        return pd.DataFrame()


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


def test_predict_evaluator_pr_auc_metric_is_skipped_for_multiclass():
    exp = FakeExperiment(is_multiclass=True)

    added = add_predict_pr_auc(exp)

    assert added is False
    assert exp.metric_added is False
