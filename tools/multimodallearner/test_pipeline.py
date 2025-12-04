from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
from plot_logic import infer_problem_type
from training_pipeline import evaluate_predictor_all_splits, fit_summary_safely

logger = logging.getLogger(__name__)


def run_autogluon_test_experiment(
    predictor,
    data_ctx: Dict[str, pd.DataFrame],
    target_column: str,
    eval_metric: Optional[str] = None,
    ag_config: Optional[dict] = None,
    problem_type: Optional[str] = None,
) -> Dict[str, object]:
    """
    Evaluate a trained predictor on train/val/test splits using prepared data_ctx.

    data_ctx is typically the context returned by ``run_autogluon_experiment``:
      {
        "train": df_train,
        "val": df_val,
        "test_internal": df_test_internal,
        "test_external": df_test_external,
        "threshold": threshold,
      }
    """
    if predictor is None:
        raise ValueError("predictor is required for evaluation.")
    if data_ctx is None:
        raise ValueError("data_ctx is required; usually from run_autogluon_experiment.")

    df_train = data_ctx.get("train")
    df_val = data_ctx.get("val")
    df_test_internal = data_ctx.get("test_internal")
    df_test_external = data_ctx.get("test_external")
    threshold = None
    if ag_config is not None:
        threshold = ag_config.get("threshold", threshold)
    threshold = data_ctx.get("threshold", threshold)

    if problem_type is None:
        # Prefer inferring from training data and predictor metadata
        base_df = df_train if df_train is not None else df_test_external
        problem_type = infer_problem_type(predictor, base_df, target_column)

    df_test_final = df_test_external if df_test_external is not None else df_test_internal
    raw_metrics, ag_by_split = evaluate_predictor_all_splits(
        predictor=predictor,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test_final,
        label_col=target_column,
        problem_type=problem_type,
        eval_metric=eval_metric,
        threshold_test=threshold,
        df_test_external=df_test_external,
    )

    summary = fit_summary_safely(predictor)

    result = {
        "problem_type": problem_type,
        "raw_metrics": raw_metrics,
        "ag_eval": ag_by_split,
        "fit_summary": summary,
    }
    logger.info("Evaluation complete; splits: %s", list(raw_metrics.keys()))
    return result
