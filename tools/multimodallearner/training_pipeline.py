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
import importlib

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
    Build hyperparameters for MultiModalPredictor.
    Handles text checkpoints for torch<2.6 and merges user overrides.
    """
    img_set = set(image_columns or [])
    text_cols = [
        c for c in df_train.columns
        if c not in img_set | {args.label_column}
        and str(df_train[c].dtype) == "object"
        and df_train[c].notna().any()
    ]

    hp = {}
    
    # Setup environment
    hp["env"] = {
        "seed": int(args.random_seed)
    }
    
    # Set eval metric through model config
    if args.eval_metric:
        # Ensure model config exists and set metric through model.metric_learning
        hp.setdefault("model", {})
        hp["model"].setdefault("metric_learning", {})["metric"] = str(args.eval_metric)

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

    # Map CLI knobs into AutoMM optimization hyperparameters when provided.
    # We set multiple common key names (nested dicts and dotted flat keys) to
    # maximize compatibility across AutoMM/AutoGluon versions.
    try:
        # Attach optimization parameters using dotted keys only. Avoid creating
        # a top-level 'optimization' dict which may not exist in the base
        # AutoMM config; also set the alternate 'optim' dotted keys to match
        # variants of saved configs across AG versions.
        if any(getattr(args, param, None) is not None for param in ["epochs", "learning_rate", "batch_size"]):
            if getattr(args, "epochs", None) is not None:
                hp["optim.max_epochs"] = int(args.epochs)
                hp["optim.epochs"] = int(args.epochs)
            if getattr(args, "learning_rate", None) is not None:
                hp["optim.learning_rate"] = float(args.learning_rate)
                hp["optim.lr"] = float(args.learning_rate)
            if getattr(args, "batch_size", None) is not None:
                hp["optim.batch_size"] = int(args.batch_size)
                hp["optim.per_device_train_batch_size"] = int(args.batch_size)

        # Also set dotted flat keys for max compatibility (e.g., 'optimization.max_epochs')
        if getattr(args, "epochs", None) is not None:
            hp["optimization.max_epochs"] = int(args.epochs)
            hp["optimization.epochs"] = int(args.epochs)
        if getattr(args, "learning_rate", None) is not None:
            hp["optimization.learning_rate"] = float(args.learning_rate)
            hp["optimization.lr"] = float(args.learning_rate)
        if getattr(args, "batch_size", None) is not None:
            hp["optimization.batch_size"] = int(args.batch_size)
            hp["optimization.per_device_train_batch_size"] = int(args.batch_size)
    except Exception:
        logger.warning("Failed to attach epochs/learning_rate/batch_size to mm_hparams; continuing without them.")

    # Map backbone selections into mm_hparams if provided
    try:
        if getattr(args, "backbone_text", None):
            # nested dict
            hp.setdefault("model.hf_text", {})["checkpoint_name"] = str(args.backbone_text)
            # dotted flat keys
            hp.setdefault("model", {})["hf_text.checkpoint_name"] = str(args.backbone_text)
            hp["model.hf_text.checkpoint_name"] = str(args.backbone_text)
        if getattr(args, "backbone_image", None):
            hp.setdefault("model.timm_image", {})["checkpoint_name"] = str(args.backbone_image)
            hp.setdefault("model", {})["timm_image.checkpoint_name"] = str(args.backbone_image)
            hp["model.timm_image.checkpoint_name"] = str(args.backbone_image)
        if getattr(args, "backbone_tabular", None):
            hp.setdefault("model", {})["tabular.backbone"] = str(args.backbone_tabular)
            hp["model.tabular.backbone"] = str(args.backbone_tabular)
    except Exception:
        logger.warning("Failed to attach backbone selections to mm_hparams; continuing without them.")

    # If AutoGluon is installed, detect version and (optionally) adapt canonical keys
    try:
        ag_mod = importlib.import_module("autogluon")
        ag_ver = getattr(ag_mod, "__version__", None)
        if ag_ver:
            ag_v = Version(str(ag_ver))
            # If needed, we could adapt key names per version here. For now, we just log.
            logger.info(f"Detected AutoGluon version: {ag_v}; applied robust hyperparameter mappings.")
    except Exception:
        # AutoGluon not present in this environment; leave mappings as-is
        pass

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
        verbosity=args.verbosity
    )
    
    # Setup hyperparameters and AG args
    hyperparameters = {}
    if args.eval_metric:
        hyperparameters["eval_metric"] = args.eval_metric
    if hyperparameters:
        tab_fit_kwargs["hyperparameters"] = hyperparameters
        
    # Set seed through AG args
    tab_fit_kwargs["ag_args_fit"] = {
        "seed": int(args.random_seed)
    }

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