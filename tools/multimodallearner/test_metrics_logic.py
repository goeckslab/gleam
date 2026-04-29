import pandas as pd
from metrics_logic import optimize_binary_threshold, resolve_threshold_metric


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
