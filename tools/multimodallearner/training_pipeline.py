from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from packaging.version import Version

from autogluon.multimodal import MultiModalPredictor

from metrics_logic import evaluate_all_transparency

logger = logging.getLogger(__name__)


# ---------------------- small utilities ----------------------
def normalize_presets(presets) -> Optional[str]:
    """
    AutoMM expects a single preset string. If multiple are provided, use the first.
    """
    if presets is None:
        return None
    if isinstance(presets, (list, tuple)):
        if len(presets) > 1:
            logger.warning(
                "MultiModalPredictor accepts a single preset. "
                f"Received {presets}; using the first: '{presets[0]}'"
            )
        return str(presets[0])
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
    inferred_text_cols = [
        c for c in df_train.columns
        if c not in img_set | {args.label_column}
        and str(df_train[c].dtype) == "object"
        and df_train[c].notna().any()
    ]
    explicit_text_cols = getattr(args, "text_columns", None)
    use_text = getattr(args, "use_text", True)
    if not use_text:
        text_cols: List[str] = []
    elif explicit_text_cols is not None:
        seen = set()
        text_cols = []
        for c in explicit_text_cols:
            if c in df_train.columns and c not in seen:
                text_cols.append(c)
                seen.add(c)
    else:
        text_cols = inferred_text_cols

    ag_version = None
    try:
        ag_mod = importlib.import_module("autogluon")
        ag_ver = getattr(ag_mod, "__version__", None)
        if ag_ver:
            ag_version = Version(str(ag_ver))
    except Exception:
        ag_mod = None

    supports_tabular_override = bool(ag_version and ag_version >= Version("1.1.0"))
    supports_fusion_override = bool(ag_version and ag_version >= Version("1.1.0"))

    def _log_missing_support(key: str) -> None:
        logger.info(
            "AutoGluon version %s does not expose '%s'; skipping override.",
            ag_version or "unknown",
            key,
        )

    hp = {}
    
    # Setup environment
    hp["env"] = {
        "seed": int(args.random_seed)
    }
    
    # Set eval metric through model config
    if args.eval_metric:
        model_block = hp.setdefault("model", {})
        model_block.setdefault("metric_learning", {})["metric"] = str(args.eval_metric)
    else:
        model_block = hp.setdefault("model", {})

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
        has_text_cols = bool(text_cols)
        has_image_cols = bool(image_columns)
        model_names_cache: Optional[List[str]] = None
        model_names_modified = False

        def _dedupe_preserve(seq: List[str]) -> List[str]:
            seen = set()
            ordered = []
            for item in seq:
                if item in seen:
                    continue
                seen.add(item)
                ordered.append(item)
            return ordered

        def _get_model_names() -> List[str]:
            nonlocal model_names_cache
            if model_names_cache is not None:
                return model_names_cache
            names = model_block.get("names")
            if isinstance(names, list):
                model_names_cache = list(names)
            else:
                model_names_cache = []
                if has_text_cols:
                    model_names_cache.append("hf_text")
                if has_image_cols:
                    model_names_cache.append("timm_image")
                model_names_cache.extend(["numerical_mlp", "categorical_mlp"])
                model_names_cache.append("fusion_mlp")
            return model_names_cache

        def _set_model_names(new_names: List[str]) -> None:
            nonlocal model_names_cache, model_names_modified
            model_names_cache = new_names
            model_names_modified = True

        if has_text_cols and getattr(args, "backbone_text", None):
            # nested dict
            hp.setdefault("model.hf_text", {})["checkpoint_name"] = str(args.backbone_text)
            # dotted flat keys
            model_block["hf_text.checkpoint_name"] = str(args.backbone_text)
            hp["model.hf_text.checkpoint_name"] = str(args.backbone_text)
        if has_image_cols and getattr(args, "backbone_image", None):
            hp.setdefault("model.timm_image", {})["checkpoint_name"] = str(args.backbone_image)
            model_block["timm_image.checkpoint_name"] = str(args.backbone_image)
            hp["model.timm_image.checkpoint_name"] = str(args.backbone_image)
        tab_choice = getattr(args, "backbone_tabular", None)
        if tab_choice:
            tab_choice = str(tab_choice)
            if supports_tabular_override:
                model_block["tabular.backbone"] = tab_choice
                hp["model.tabular.backbone"] = tab_choice
            else:
                _log_missing_support("model.tabular.backbone")
            tabular_module_map = {
                "ft_transformer": ["ft_transformer"],
                "numerical_mlp": ["numerical_mlp"],
                "categorical_mlp": ["categorical_mlp"],
            }
            desired_modules = tabular_module_map.get(tab_choice, [])
            if desired_modules:
                names = _get_model_names()
                filtered = [n for n in names if n not in {"numerical_mlp", "categorical_mlp", "ft_transformer"}]
                filtered.extend(desired_modules)
                _set_model_names(_dedupe_preserve(filtered))
        fusion_choice = getattr(args, "backbone_fusion", None)
        if fusion_choice:
            fusion_choice = str(fusion_choice)
            if supports_fusion_override:
                model_block["fusion.backbone"] = fusion_choice
                model_block["fusion"] = fusion_choice
                hp["model.fusion.backbone"] = fusion_choice
                hp["model.fusion"] = fusion_choice
                hp["model.fusion_backbone"] = fusion_choice
            else:
                _log_missing_support("model.fusion.backbone")
            names = _get_model_names()
            filtered = [n for n in names if n not in ("fusion_mlp", "fusion_transformer")]
            filtered.append(fusion_choice)
            _set_model_names(_dedupe_preserve(filtered))

        if model_names_modified and model_names_cache is not None:
            model_block["names"] = model_names_cache
    except Exception:
        logger.warning("Failed to attach backbone selections to mm_hparams; continuing without them.")

    # If AutoGluon is installed, detect version and (optionally) adapt canonical keys
    if ag_version:
        logger.info(f"Detected AutoGluon version: {ag_version}; applied robust hyperparameter mappings.")

    return hp


def train_predictor(
    args,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    image_columns: Optional[List[str]],
    mm_hparams: dict,
):
    """
    Train a MultiModalPredictor, honoring common knobs (presets, eval_metric, etc.).
    """
    logger.info("Starting AutoGluon MultiModal training...")
    predictor = MultiModalPredictor(label=args.label_column, path=None)
    column_types = {c: "image_path" for c in (image_columns or [])}

    mm_fit_kwargs = dict(
        train_data=df_train,
        time_limit=args.time_limit,
        seed=int(args.random_seed),
        hyperparameters=mm_hparams,
    )
    if df_val is not None and not df_val.empty:
        mm_fit_kwargs["tuning_data"] = df_val
    if column_types:
        mm_fit_kwargs["column_types"] = column_types

    preset_mm = normalize_presets(args.presets)
    if preset_mm is not None:
        mm_fit_kwargs["presets"] = preset_mm

    predictor.fit(**mm_fit_kwargs)
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
