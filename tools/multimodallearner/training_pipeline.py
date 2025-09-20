from __future__ import annotations

import contextlib
import io
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from packaging.version import Version

from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor

from metrics_logic import evaluate_all_transparency

logger = logging.getLogger(__name__)


# ---------------------- small utilities ----------------------
def normalize_presets(presets, for_multimodal: bool) -> Optional[str]:
    """
    AutoMM expects a single string preset; Tabular accepts a string as well.
    If a list is provided:
      - MultiModal: use the first and warn.
      - Tabular: join with spaces ("best_quality optimize_for_deployment").
    """
    if presets is None:
        return None
    if isinstance(presets, (list, tuple)):
        if for_multimodal:
            if len(presets) > 1:
                logger.warning(
                    "MultiModalPredictor accepts a single preset. "
                    f"Received {presets}; using the first: '{presets[0]}'"
                )
            return str(presets[0])
        # Tabular path: join tokens into one string
        return " ".join(str(p) for p in presets)
    return str(presets)

def load_user_hparams(hp_arg: Optional[str]) -> dict:
    """Parse --hyperparameters (inline JSON or path to .json)."""
    if not hp_arg:
        return {}
    try:
        s = hp_arg.strip()
        if s.startswith("{"):
            return json.loads(s)
        with open(s, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not parse --hyperparameters: {e}. Ignoring.")
        return {}


def deep_update(dst: dict, src: dict) -> dict:
    """Recursive dict update (src overrides dst)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Silence noisy prints from AG internals (fit_summary)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def ag_evaluate_safely(predictor, df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Call predictor.evaluate and normalize the output to a dict.
    """
    try:
        res = predictor.evaluate(df, metrics=metrics)
    except TypeError:
        if metrics and len(metrics) == 1:
            res = predictor.evaluate(df, metrics[0])
        else:
            res = predictor.evaluate(df)
    if isinstance(res, (int, float, np.floating)):
        name = (metrics[0] if metrics else "metric")
        return {name: float(res)}
    if isinstance(res, dict):
        return {k: float(v) for k, v in res.items()}
    return {"metric": float(res)}


# ---------------------- hparams & training ----------------------
def build_mm_hparams(args, df_train: pd.DataFrame, image_columns: Optional[List[str]]) -> dict:
    """
    Start with seed, optionally force a safe HF text checkpoint on torch<2.6,
    then merge user overrides. If --eval_metric is given, set optimization.metric.
    """
    img_set = set(image_columns or [])
    text_cols = [
        c for c in df_train.columns
        if c not in img_set | {args.label_column}
        and str(df_train[c].dtype) == "object"
        and df_train[c].notna().any()
    ]

    hp = {"optimization": {"seed": int(args.random_seed)}}

    # Plug in the requested eval metric for AutoMM (accepted via hyperparameters)
    if args.eval_metric:
        # AutoMM expects the name as a string; 'roc_auc' is valid for binary.
        hp["optimization"]["metric"] = str(args.eval_metric)

    if text_cols and Version(torch.__version__) < Version("2.6"):
        safe_ckpt = "distilbert-base-uncased"
        logger.warning(f"Forcing HF text checkpoint with safetensors: {safe_ckpt}")
        hp["model.hf_text.checkpoint_name"] = safe_ckpt
        hp.setdefault(
            "model.names",
            ["hf_text", "timm_image", "numerical_mlp", "categorical_mlp", "fusion_mlp"],
        )

    user_hp = load_user_hparams(args.hyperparameters)
    hp = deep_update(hp, user_hp)
    return hp


def train_predictor(
    args,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    image_columns: Optional[List[str]],
    mm_hparams: dict,
):
    """
    Train either MultiModalPredictor (if image columns) or TabularPredictor (else),
    honoring cheat-sheet knobs (presets, eval_metric, bagging/stacking, etc.).
    """
    presets_arg = args.presets if args.presets else None

    if image_columns:
        logger.info("Starting AutoGluon MultiModal training...")
        column_types = {c: "image_path" for c in image_columns}
        predictor = MultiModalPredictor(label=args.label_column, path=None)

        # NOTE: AutoMM does not accept eval_metric / verbosity as kwargs.
        mm_fit_kwargs = dict(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
            seed=int(args.random_seed),
            column_types=column_types,
            hyperparameters=mm_hparams,
        )
        preset_mm = normalize_presets(args.presets, for_multimodal=True)
        if preset_mm is not None:
            mm_fit_kwargs["presets"] = preset_mm

        predictor.fit(**mm_fit_kwargs)
        return predictor

# --- Tabular ---
    logger.info("Starting AutoGluon Tabular training...")
    predictor = TabularPredictor(label=args.label_column, path=None)

    tab_fit_kwargs = dict(
        train_data=df_train,
        tuning_data=df_val,
        time_limit=args.time_limit,
        eval_metric=args.eval_metric,
        verbosity=args.verbosity,
        seed=int(args.random_seed),
    )

    preset_tab = normalize_presets(args.presets, for_multimodal=False)
    if preset_tab is not None:
        tab_fit_kwargs["presets"] = preset_tab

    ag_args_fit = {}
    if args.num_bag_folds is not None:
        ag_args_fit["num_bag_folds"] = int(args.num_bag_folds)
    if args.num_stack_levels is not None:
        ag_args_fit["num_stack_levels"] = int(args.num_stack_levels)
    if ag_args_fit:
        tab_fit_kwargs["ag_args_fit"] = ag_args_fit

    if args.excluded_model_types:
        tab_fit_kwargs["excluded_model_types"] = args.excluded_model_types

    predictor.fit(**tab_fit_kwargs)

    if args.refit_full:
        try:
            logger.info("Refitting best model on all (train+val) data (refit_full=True)...")
            predictor.refit_full()
        except Exception as e:
            logger.warning(f"refit_full failed: {e}")

    return predictor


# ---------------------- evaluation ----------------------
def evaluate_predictor_all_splits(
    predictor,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str,
    problem_type: str,
    eval_metric: Optional[str],
    threshold_test: Optional[float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Returns (raw_metrics, ag_scores_by_split)
      - raw_metrics: our transparent suite (threshold applied to Test only inside metrics_logic)
      - ag_scores_by_split: AutoGluon's evaluate() per split for the chosen eval_metric (or default)
    """
    metrics_req = [eval_metric] if eval_metric else None
    ag_scores_train = ag_evaluate_safely(predictor, df_train, metrics=metrics_req)
    ag_scores_val   = ag_evaluate_safely(predictor, df_val,   metrics=metrics_req)
    ag_scores_test  = ag_evaluate_safely(predictor, df_test,  metrics=metrics_req)

    # Transparent suite (threshold on Test only handled inside metrics_logic)
    _, raw_metrics = evaluate_all_transparency(
        predictor=predictor,
        train_df=df_train,
        val_df=df_val,
        test_df=df_test,
        target_col=label_col,
        problem_type=problem_type,
        threshold=threshold_test,
    )

    ag_by_split = {
        "Train": ag_scores_train,
        "Validation": ag_scores_val,
        "Test": ag_scores_test,
    }
    return raw_metrics, ag_by_split


def fit_summary_safely(predictor) -> Optional[dict]:
    """Get fit summary without printing misleading one-liners."""
    with suppress_stdout_stderr():
        try:
            return predictor.fit_summary()
        except Exception:
            return None