from report_utils import _resolve_report_eval_metric


class Args:
    def __init__(self, eval_metric):
        self.eval_metric = eval_metric


class PredictorWithMetric:
    eval_metric = "roc_auc"


class MetricObject:
    name = "balanced_accuracy"


class PredictorWithMetricObject:
    eval_metric = MetricObject()


class PredictorWithoutMetric:
    eval_metric = None


def test_resolve_report_eval_metric_uses_predictor_metric_for_auto():
    metric = _resolve_report_eval_metric(Args("auto"), PredictorWithMetric(), {})

    assert metric == "roc_auc"


def test_resolve_report_eval_metric_keeps_explicit_user_metric():
    metric = _resolve_report_eval_metric(Args("f1_macro"), PredictorWithMetric(), {})

    assert metric == "f1_macro"


def test_resolve_report_eval_metric_handles_scorer_objects():
    metric = _resolve_report_eval_metric(Args("auto"), PredictorWithMetricObject(), {})

    assert metric == "balanced_accuracy"


def test_resolve_report_eval_metric_falls_back_to_autogluon_evaluate_key():
    eval_results = {"ag_eval": {"Validation": {"accuracy": 0.75}}}

    metric = _resolve_report_eval_metric(Args("auto"), PredictorWithoutMetric(), eval_results)

    assert metric == "accuracy"
