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


def _load_tool_modules():
    from base_model_trainer import _add_pr_auc_metric_if_supported, BaseModelTrainer
    from classification_metrics import (
        labels_in_metric_order,
        labels_in_sample_order,
        weighted_ovr_pr_auc,
    )
    from feature_importance import FeatureImportanceAnalyzer
    from pycaret_classification import (
        _should_skip_pycaret_plot,
        ClassificationModelTrainer,
    )
    from pycaret_predict import _add_pr_auc_metric_if_supported as add_predict_pr_auc
    from pycaret_regression import RegressionModelTrainer

    return (
        _add_pr_auc_metric_if_supported,
        BaseModelTrainer,
        labels_in_metric_order,
        labels_in_sample_order,
        weighted_ovr_pr_auc,
        FeatureImportanceAnalyzer,
        _should_skip_pycaret_plot,
        ClassificationModelTrainer,
        add_predict_pr_auc,
        RegressionModelTrainer,
    )


(
    _add_pr_auc_metric_if_supported,
    BaseModelTrainer,
    labels_in_metric_order,
    labels_in_sample_order,
    weighted_ovr_pr_auc,
    FeatureImportanceAnalyzer,
    _should_skip_pycaret_plot,
    ClassificationModelTrainer,
    add_predict_pr_auc,
    RegressionModelTrainer,
) = _load_tool_modules()


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


class FakeModelWithoutProba:
    def predict(self, X=None):
        n_rows = len(X) if X is not None else 0
        return np.resize(np.asarray([1, 2, 3, 4, 5]), n_rows)


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
        self.X_test_transformed = self.X_test
        self.y_test_transformed = self.y_test
        self.pipeline = None

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


def test_multiclass_plot_generation_does_not_add_binary_proba_patch():
    exp = FakePlotExperiment()
    model = FakeModelWithoutProba()
    trainer = object.__new__(ClassificationModelTrainer)
    trainer.best_model = model
    trainer.exp = exp
    trainer.plots = {}

    trainer.generate_plots()

    assert not hasattr(model, "predict_proba")


def test_multiclass_explainer_failure_keeps_custom_plots(monkeypatch):
    explainerdashboard = types.ModuleType("explainerdashboard")

    class RaisingClassifierExplainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no multiclass probabilities")

    explainerdashboard.ClassifierExplainer = RaisingClassifierExplainer
    monkeypatch.setitem(sys.modules, "explainerdashboard", explainerdashboard)

    exp = FakeExperiment(is_multiclass=True)
    trainer = object.__new__(ClassificationModelTrainer)
    trainer.best_model = FakeModelWithoutProba()
    trainer.exp = exp
    trainer.task_type = "classification"
    trainer.target = "AGE"
    trainer.test_data = None
    trainer.random_seed = 42
    trainer.plot_feature_names = []
    trainer.explainer_plots = {}
    trainer._limit_explainer_data = lambda X, y, **kwargs: (X, y)

    trainer.generate_plots_explainer()

    assert "class_report" in trainer.explainer_plots
    assert "confusion_matrix" in trainer.explainer_plots


def test_weighted_multiclass_pr_auc_uses_probability_column_label_order():
    y_true = pd.Series(["beta", "alpha", "gamma", "beta", "alpha", "gamma"])
    class_order = ["gamma", "alpha", "beta"]
    y_score = np.array(
        [
            [0.05, 0.10, 0.85],
            [0.10, 0.80, 0.10],
            [0.90, 0.05, 0.05],
            [0.05, 0.20, 0.75],
            [0.15, 0.70, 0.15],
            [0.80, 0.10, 0.10],
        ]
    )

    assert weighted_ovr_pr_auc(y_true, y_score, labels=class_order) == 1.0
    assert weighted_ovr_pr_auc(
        y_true,
        y_score,
        labels=labels_in_sample_order(y_true),
    ) < 1.0


def test_binary_pr_auc_default_label_order_is_metric_stable():
    y_true = pd.Series(["yes", "no", "yes", "no"])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])

    assert labels_in_sample_order(y_true) == ["yes", "no"]
    assert labels_in_metric_order(y_true) == ["no", "yes"]
    assert weighted_ovr_pr_auc(y_true, y_score) == 1.0


def test_report_confusion_matrix_preserves_sample_label_order():
    trainer = object.__new__(ClassificationModelTrainer)
    trainer.task_type = "classification"
    trainer.target = "target"
    trainer.exp = FakeExperiment(is_multiclass=True)

    labels = ["beta", "alpha", "gamma"]
    fig = trainer._build_confusion_matrix_fig(
        pd.Series(["beta", "alpha", "gamma"]),
        pd.Series(["beta", "gamma", "gamma"]),
        labels,
    )

    assert fig.data[0].x == tuple(f"Pred {label}" for label in labels)
    assert fig.data[0].y == tuple(f"True {label}" for label in labels)


def test_labeled_scatter_preserves_sample_label_order():
    import plotly.graph_objects as go

    trainer = object.__new__(ClassificationModelTrainer)
    fig = go.Figure()

    trainer._plot_labeled_scatter(
        fig,
        x=np.array([0, 1, 2]),
        y=np.array([0, 1, 2]),
        labels=pd.Series(["beta", "alpha", "gamma"]),
    )

    assert [trace.name for trace in fig.data] == ["beta", "alpha", "gamma"]


def test_multiclass_custom_curves_use_model_class_order_and_display_labels():
    trainer = object.__new__(ClassificationModelTrainer)
    trainer.task_type = "classification"
    trainer.target = "target"
    trainer.exp = FakeExperiment(is_multiclass=True)
    trainer._decode_labels_for_display = lambda values: values

    y_true = pd.Series(["beta", "alpha", "gamma", "beta", "alpha", "gamma"])
    score_labels = ["gamma", "alpha", "beta"]
    y_score = np.array(
        [
            [0.05, 0.10, 0.85],
            [0.10, 0.80, 0.10],
            [0.90, 0.05, 0.05],
            [0.05, 0.20, 0.75],
            [0.15, 0.70, 0.15],
            [0.80, 0.10, 0.10],
        ]
    )

    roc_fig = trainer._build_multiclass_roc_fig(y_true, y_score, score_labels)
    pr_fig = trainer._build_multiclass_precision_recall_fig(
        y_true,
        y_score,
        score_labels,
    )

    assert [trace.name for trace in roc_fig.data[:3]] == [
        "gamma (AUC=1.000)",
        "alpha (AUC=1.000)",
        "beta (AUC=1.000)",
    ]
    assert [trace.name for trace in pr_fig.data] == [
        "gamma (AUC=1.000)",
        "alpha (AUC=1.000)",
        "beta (AUC=1.000)",
    ]


def test_regression_permutation_importance_uses_supported_explainer_api(monkeypatch):
    calls = []
    explainerdashboard = types.ModuleType("explainerdashboard")

    class FakeRegressionExplainer:
        X = pd.DataFrame({"feature": [0, 1]})
        onehot_cols = []

        def __init__(self, *args, **kwargs):
            pass

        def plot_importances(self, **kwargs):
            calls.append(kwargs)
            return "importance"

        def plot_pdp(self, feature):
            return f"pdp-{feature}"

        def plot_predicted_vs_actual(self):
            return "predicted-vs-actual"

        def plot_residuals(self):
            return "residuals"

        def plot_residuals_vs_feature(self, feature):
            return f"residuals-{feature}"

    explainerdashboard.RegressionExplainer = FakeRegressionExplainer
    monkeypatch.setitem(sys.modules, "explainerdashboard", explainerdashboard)

    trainer = object.__new__(RegressionModelTrainer)
    trainer.best_model = object()
    trainer.exp = types.SimpleNamespace(
        X_test_transformed=pd.DataFrame({"feature": [0, 1]}),
        y_test_transformed=pd.Series([0.0, 1.0]),
    )
    trainer.random_seed = 42
    trainer.plot_feature_names = []
    trainer.explainer_plots = {}
    trainer.explainer_scope = types.SimpleNamespace(
        total_features=1,
        feature_cap=30,
    )
    trainer.explainer_dashboard_importance_skipped = False
    trainer._limit_explainer_data = lambda X, y, **kwargs: (X, y)
    trainer._explainer_feature_count_exceeds_cap = lambda: False

    trainer.generate_plots_explainer()
    trainer.explainer_plots["shap_perm"]()

    assert {"kind": "permutation"} in calls


def test_tree_plot_failure_does_not_block_html_report(caplog):
    calls = []
    trainer = object.__new__(BaseModelTrainer)
    trainer.load_data = lambda: calls.append("load_data")
    trainer.setup_pycaret = lambda: calls.append("setup_pycaret")
    trainer.train_model = lambda: calls.append("train_model")
    trainer.save_model = lambda: calls.append("save_model")
    trainer.generate_plots = lambda: calls.append("generate_plots")
    trainer.generate_plots_explainer = lambda: calls.append("generate_plots_explainer")
    trainer.save_html_report = lambda: calls.append("save_html_report")

    def _raise_tree_error():
        calls.append("generate_tree_plots")
        raise RuntimeError("dot failed")

    trainer.generate_tree_plots = _raise_tree_error

    BaseModelTrainer.run(trainer)

    assert calls == [
        "load_data",
        "setup_pycaret",
        "train_model",
        "save_model",
        "generate_plots",
        "generate_plots_explainer",
        "generate_tree_plots",
        "save_html_report",
    ]
    assert "Tree plots skipped: dot failed" in caplog.text


class FakeShapExplanation:
    def __init__(self, shape, selected=None):
        self.shape = shape
        self.selected = [] if selected is None else selected

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            class_idx = key[1]
            return FakeShapExplanation(self.shape[:2], self.selected + [class_idx])
        raise AssertionError(f"Unexpected SHAP slice: {key!r}")


class FakeShapExplainer:
    def __init__(self, explanation):
        self.explanation = explanation

    def __call__(self, X):
        return self.explanation


def _build_shap_analyzer(tmp_path, task_type, model, explanation, max_features=1):
    analyzer = object.__new__(FeatureImportanceAnalyzer)
    analyzer.task_type = task_type
    analyzer.best_model = model
    analyzer.exp = types.SimpleNamespace(pipeline=None)
    analyzer.output_dir = str(tmp_path)
    analyzer.plots = {}
    analyzer.plot_titles = {}
    analyzer.max_shap_rows = None
    analyzer.max_plot_features = max_features
    analyzer.polynomial_features = False
    analyzer._get_transformed_frame = lambda model, prefer_test=True: pd.DataFrame(
        {
            "f1": [0.0, 1.0],
            "f2": [2.0, 3.0],
            "f3": [4.0, 5.0],
        }
    )
    analyzer._choose_shap_explainer = lambda model, bg, predict_fn: (
        FakeShapExplainer(explanation),
        "fake",
        False,
    )
    return analyzer


def test_custom_shap_binary_plots_positive_class_only_with_display_cap(
    monkeypatch,
    tmp_path,
):
    import feature_importance

    beeswarm_calls = []
    monkeypatch.setattr(
        feature_importance.shap,
        "plots",
        types.SimpleNamespace(
            beeswarm=lambda explanation, **kwargs: beeswarm_calls.append(
                (explanation.shape, explanation.selected, kwargs)
            )
        ),
        raising=False,
    )

    class FakeBinaryModel:
        classes_ = ["control", "case"]

        def predict_proba(self, X):
            return np.ones((len(X), 2)) * 0.5

    explanation = FakeShapExplanation((2, 3, 2))
    analyzer = _build_shap_analyzer(
        tmp_path,
        "classification",
        FakeBinaryModel(),
        explanation,
        max_features=1,
    )

    analyzer.save_shap_values()

    assert beeswarm_calls == [((2, 3), [1], {"max_display": 1, "show": False})]
    assert list(analyzer.plots) == ["shap_summary_class_case"]
    assert "class case" in analyzer.plot_titles["shap_summary_class_case"]
    assert analyzer.shap_used_features == 3


def test_custom_shap_multiclass_plots_each_model_class(monkeypatch, tmp_path):
    import feature_importance

    beeswarm_calls = []
    monkeypatch.setattr(
        feature_importance.shap,
        "plots",
        types.SimpleNamespace(
            beeswarm=lambda explanation, **kwargs: beeswarm_calls.append(
                (explanation.shape, explanation.selected, kwargs)
            )
        ),
        raising=False,
    )

    class FakeMulticlassModel:
        classes_ = ["gamma", "alpha", "beta"]

        def predict_proba(self, X):
            return np.ones((len(X), 3)) / 3

    explanation = FakeShapExplanation((2, 3, 3))
    analyzer = _build_shap_analyzer(
        tmp_path,
        "classification",
        FakeMulticlassModel(),
        explanation,
        max_features=2,
    )

    analyzer.save_shap_values()

    assert [call[1] for call in beeswarm_calls] == [[0], [1], [2]]
    assert list(analyzer.plots) == [
        "shap_summary_class_gamma",
        "shap_summary_class_alpha",
        "shap_summary_class_beta",
    ]
    assert all(call[2]["max_display"] == 2 for call in beeswarm_calls)


def test_custom_shap_regression_plots_single_summary(monkeypatch, tmp_path):
    import feature_importance

    beeswarm_calls = []
    monkeypatch.setattr(
        feature_importance.shap,
        "plots",
        types.SimpleNamespace(
            beeswarm=lambda explanation, **kwargs: beeswarm_calls.append(
                (explanation.shape, explanation.selected, kwargs)
            )
        ),
        raising=False,
    )

    class FakeRegressionModel:
        def predict(self, X):
            return np.zeros(len(X))

    explanation = FakeShapExplanation((2, 3))
    analyzer = _build_shap_analyzer(
        tmp_path,
        "regression",
        FakeRegressionModel(),
        explanation,
        max_features=2,
    )

    analyzer.save_shap_values()

    assert beeswarm_calls == [((2, 3), [], {"max_display": 2, "show": False})]
    assert list(analyzer.plots) == ["shap_summary"]
    assert analyzer.shap_used_features == 3
