import pandas as pd
from metrics_logic import evaluate_all_transparency, optimize_binary_threshold, resolve_threshold_metric


class FakeBinaryPredictor:
    class_labels = [0, 1]

    def __init__(self, positive_scores):
        self.positive_scores = positive_scores

    def predict_proba(self, features):
        return pd.DataFrame(
            {
                0: [1.0 - score for score in self.positive_scores],
                1: self.positive_scores,
            }
        )


class FeatureScoreBinaryPredictor:
    class_labels = [0, 1]

    def predict_proba(self, features):
        scores = features["feature"].astype(float).reset_index(drop=True)
        return pd.DataFrame({0: 1.0 - scores, 1: scores})

    def predict(self, features):
        scores = self.predict_proba(features)[1]
        return pd.Series((scores >= 0.5).astype(int))


def test_resolve_threshold_metric_uses_threshold_sensitive_eval_metric():
    assert resolve_threshold_metric("auto", eval_metric="balanced_accuracy") == "balanced_accuracy"


def test_resolve_threshold_metric_falls_back_to_f1_for_ranking_metric():
    assert resolve_threshold_metric("auto", eval_metric="roc_auc") == "f1"


def test_optimize_binary_threshold_uses_validation_probabilities():
    df = pd.DataFrame({"feature": [0, 1, 2, 3], "label": [0, 0, 1, 1]})
    predictor = FakeBinaryPredictor([0.1, 0.4, 0.35, 0.8])

    result = optimize_binary_threshold(
        predictor=predictor,
        df=df,
        target_col="label",
        threshold_metric="f1",
    )

    assert result["threshold"] == 0.35
    assert result["threshold_metric"] == "f1"
    assert result["threshold_source"] == "Optimized on validation split"
    assert round(result["threshold_metric_value"], 3) == 0.8


def test_evaluate_all_transparency_applies_binary_threshold_to_all_splits():
    df = pd.DataFrame({"feature": [0.1, 0.2, 0.35, 0.4], "label": [0, 0, 1, 1]})

    _, raw_metrics, _ = evaluate_all_transparency(
        predictor=FeatureScoreBinaryPredictor(),
        train_df=df,
        val_df=df,
        test_df=df,
        target_col="label",
        problem_type="binary",
        threshold=0.3,
    )

    for split in ["Train", "Validation", "Test"]:
        assert raw_metrics[split]["Precision"] == 1.0
        assert raw_metrics[split]["Recall_(Sensitivity/TPR)"] == 1.0
        assert raw_metrics[split]["F1-Score"] == 1.0
