from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from autogluon.multimodal import MultiModalPredictor
from metrics_logic import compute_metrics_for_split, evaluate_all_transparency
from packaging.version import Version

logger = logging.getLogger(__name__)

_LOW_SHM_BYTES = 1 << 30  # 1 GiB


def _get_env_int(keys: List[str]) -> Optional[int]:
    for key in keys:
        if key not in os.environ:
            continue
        raw = os.environ.get(key)
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-integer %s=%s", key, raw)
    return None


def _get_shm_bytes() -> Optional[int]:
    try:
        stat = os.statvfs("/dev/shm")
    except Exception:
        return None
    return int(stat.f_frsize * stat.f_blocks)


def _resolve_num_workers(
    explicit_value: Optional[int],
    env_keys: List[str],
    label: str,
    shm_bytes: Optional[int],
) -> Optional[int]:
    if explicit_value is not None:
        return int(explicit_value)
    env_val = _get_env_int(env_keys)
    if env_val is not None:
        return env_val
    if shm_bytes is not None and shm_bytes < _LOW_SHM_BYTES:
        logger.warning(
            "Detected small /dev/shm (%.1f MB); setting %s num_workers=0 to avoid DataLoader shm errors.",
            shm_bytes / (1024 * 1024),
            label,
        )
        return 0
    return None

# ---------------------- small utilities ----------------------


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


def ag_evaluate_safely(predictor, df: Optional[pd.DataFrame], metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Call predictor.evaluate and normalize the output to a dict.
    """
    if df is None or len(df) == 0:
        return {}
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
    inferred_text_cols = [
        c for c in df_train.columns
        if c != args.label_column
        and str(df_train[c].dtype) == "object"
        and df_train[c].notna().any()
    ]
    text_cols = inferred_text_cols

    ag_version = None
    try:
        ag_mod = importlib.import_module("autogluon")
        ag_ver = getattr(ag_mod, "__version__", None)
        if ag_ver:
            ag_version = Version(str(ag_ver))
    except Exception:
        ag_mod = None

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
    model_block = hp.setdefault("model", {})
    if args.eval_metric:
        model_block.setdefault("metric_learning", {})["metric"] = str(args.eval_metric)

    if text_cols and Version(torch.__version__) < Version("2.6"):
        safe_ckpt = "distilbert-base-uncased"
        logger.warning(f"Forcing HF text checkpoint with safetensors: {safe_ckpt}")
        hp["model.hf_text.checkpoint_name"] = safe_ckpt
        hp.setdefault(
            "model.names",
            ["hf_text", "timm_image", "numerical_mlp", "categorical_mlp", "fusion_mlp"],
        )

    def _is_valid_hp_dict(d) -> bool:
        if not isinstance(d, dict):
            logger.warning("User-supplied hyperparameters must be a dict; received %s", type(d).__name__)
            return False
        return True

    user_hp = args.hyperparameters if isinstance(args.hyperparameters, dict) else load_user_hparams(args.hyperparameters)
    if user_hp and _is_valid_hp_dict(user_hp):
        hp = deep_update(hp, user_hp)

    # Map CLI knobs into AutoMM optimization hyperparameters when provided.
    # We set multiple common key names (nested dicts and dotted flat keys) to
    # maximize compatibility across AutoMM/AutoGluon versions.
    try:
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
        has_image_cols = False
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
            text_choice = str(args.backbone_text)
            model_block.setdefault("hf_text", {})["checkpoint_name"] = text_choice
            hp["model.hf_text.checkpoint_name"] = text_choice
        if has_image_cols and getattr(args, "backbone_image", None):
            image_choice = str(args.backbone_image)
            model_block.setdefault("timm_image", {})["checkpoint_name"] = image_choice
            hp["model.timm_image.checkpoint_name"] = image_choice
        if model_names_modified and model_names_cache is not None:
            model_block["names"] = model_names_cache
    except Exception:
        logger.warning("Failed to attach backbone selections to mm_hparams; continuing without them.")

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
    column_types = {}

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

    preset_mm = getattr(args, "presets", None)
    if preset_mm is None:
        preset_mm = getattr(args, "preset", None)
    if preset_mm is not None:
        mm_fit_kwargs["presets"] = preset_mm

    predictor.fit(**mm_fit_kwargs)
    return predictor


# ---------------------- evaluation ----------------------
def evaluate_predictor_all_splits(
    predictor,
    df_train: Optional[pd.DataFrame],
    df_val: Optional[pd.DataFrame],
    df_test: Optional[pd.DataFrame],
    label_col: str,
    problem_type: str,
    eval_metric: Optional[str],
    threshold_test: Optional[float],
    df_test_external: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Returns (raw_metrics, ag_scores_by_split)
      - raw_metrics: our transparent suite (threshold applied to Test/External Test only inside metrics_logic)
      - ag_scores_by_split: AutoGluon's evaluate() per split for the chosen eval_metric (or default)
    """
    metrics_req = None if (eval_metric is None or str(eval_metric).lower() == "auto") else [eval_metric]
    ag_by_split: Dict[str, Dict[str, float]] = {}

    if df_train is not None and len(df_train):
        ag_by_split["Train"] = ag_evaluate_safely(predictor, df_train, metrics=metrics_req)
    if df_val is not None and len(df_val):
        ag_by_split["Validation"] = ag_evaluate_safely(predictor, df_val, metrics=metrics_req)

    df_test_effective = df_test_external if df_test_external is not None else df_test
    if df_test_effective is not None and len(df_test_effective):
        ag_by_split["Test"] = ag_evaluate_safely(predictor, df_test_effective, metrics=metrics_req)

    # Transparent suite (threshold on Test handled inside metrics_logic)
    _, raw_metrics = evaluate_all_transparency(
        predictor=predictor,
        train_df=df_train,
        val_df=df_val,
        test_df=df_test_effective,
        target_col=label_col,
        problem_type=problem_type,
        threshold=threshold_test,
    )

    if df_test_external is not None and df_test_external is not df_test and len(df_test_external):
        raw_metrics["Test (external)"] = compute_metrics_for_split(
            predictor, df_test_external, label_col, problem_type, threshold=threshold_test
        )
        ag_by_split["Test (external)"] = ag_evaluate_safely(predictor, df_test_external, metrics=metrics_req)

    return raw_metrics, ag_by_split


def fit_summary_safely(predictor) -> Optional[dict]:
    """Get fit summary without printing misleading one-liners."""
    with suppress_stdout_stderr():
        try:
            return predictor.fit_summary()
        except Exception:
            return None


# ---------------------- image helpers ----------------------
_PLACEHOLDER_PATH = None


def _create_placeholder() -> str:
    global _PLACEHOLDER_PATH
    if _PLACEHOLDER_PATH and os.path.exists(_PLACEHOLDER_PATH):
        return _PLACEHOLDER_PATH

    dir_ = Path(tempfile.mkdtemp(prefix="ag_placeholder_"))
    file_ = dir_ / f"placeholder_{uuid.uuid4().hex}.png"

    try:
        from PIL import Image
        Image.new("RGB", (64, 64), (180, 180, 180)).save(file_)
    except Exception:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.imsave(file_, np.full((64, 64, 3), 180, dtype=np.uint8))
        plt.close("all")

    _PLACEHOLDER_PATH = str(file_)
    logger.info(f"Placeholder image created: {file_}")
    return _PLACEHOLDER_PATH


def _is_valid_path(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    return s and os.path.isfile(s)


def handle_missing_images(
    df: pd.DataFrame,
    image_columns: List[str],
    strategy: str = "false",
) -> pd.DataFrame:
    if not image_columns or df.empty:
        return df

    remove = str(strategy).lower() == "true"
    masks = [~df[col].apply(_is_valid_path) for col in image_columns if col in df.columns]
    if not masks:
        return df

    any_missing = pd.concat(masks, axis=1).any(axis=1)
    n_missing = int(any_missing.sum())

    if n_missing == 0:
        return df

    if remove:
        result = df[~any_missing].reset_index(drop=True)
        logger.info(f"Dropped {n_missing} rows with missing images → {len(result)} remain")
    else:
        placeholder = _create_placeholder()
        result = df.copy()
        for col in image_columns:
            if col in result.columns:
                result.loc[~result[col].apply(_is_valid_path), col] = placeholder
        logger.info(f"Filled {n_missing} missing images with placeholder")

    return result


# ---------------------- AutoGluon config helpers ----------------------
def autogluon_hyperparameters(
    threshold,
    time_limit,
    random_seed,
    epochs,
    learning_rate,
    batch_size,
    num_workers,
    num_workers_evaluation,
    backbone_image,
    backbone_text,
    preset,
    eval_metric,
    hyperparameters,
):
    """
    Build a MultiModalPredictor configuration (fit kwargs + hyperparameters) from CLI inputs.
    The returned dict separates what should be passed to predictor.fit (under ``fit``)
    from the model/optimization configuration (under ``hyperparameters``). Threshold is
    preserved for downstream evaluation but not passed into AutoGluon directly.
    """

    def _prune_empty(d: dict) -> dict:
        cleaned = {}
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                nested = _prune_empty(v)
                if nested:
                    cleaned[k] = nested
            elif v is not None:
                cleaned[k] = v
        return cleaned

    # Base hyperparameters following the structure described in the AutoGluon
    # customization guide (env / optimization / model).
    env_cfg = {}
    if random_seed is not None:
        env_cfg["seed"] = int(random_seed)
    if batch_size is not None:
        env_cfg["per_gpu_batch_size"] = int(batch_size)
    shm_bytes = _get_shm_bytes()
    resolved_num_workers = _resolve_num_workers(
        num_workers,
        ["AG_MM_NUM_WORKERS", "AG_NUM_WORKERS", "AUTOMM_NUM_WORKERS"],
        "training",
        shm_bytes,
    )
    resolved_num_workers_eval = _resolve_num_workers(
        num_workers_evaluation,
        ["AG_MM_NUM_WORKERS_EVAL", "AG_MM_NUM_WORKERS_EVALUATION", "AUTOMM_NUM_WORKERS_EVAL"],
        "evaluation",
        shm_bytes,
    )
    if resolved_num_workers_eval is None and resolved_num_workers is not None:
        resolved_num_workers_eval = resolved_num_workers
    if resolved_num_workers is not None:
        env_cfg["num_workers"] = int(resolved_num_workers)
    if resolved_num_workers_eval is not None:
        env_cfg["num_workers_evaluation"] = int(resolved_num_workers_eval)

    optim_cfg = {}
    if epochs is not None:
        optim_cfg["max_epochs"] = int(epochs)
    if learning_rate is not None:
        optim_cfg["learning_rate"] = float(learning_rate)
    if batch_size is not None:
        bs = int(batch_size)
        optim_cfg["per_device_train_batch_size"] = bs
        optim_cfg["train_batch_size"] = bs

    model_cfg = {}
    if eval_metric:
        model_cfg.setdefault("metric_learning", {})["metric"] = str(eval_metric)
    if backbone_image:
        model_cfg.setdefault("timm_image", {})["checkpoint_name"] = str(backbone_image)
    if backbone_text:
        model_cfg.setdefault("hf_text", {})["checkpoint_name"] = str(backbone_text)

    hp = {
        "env": env_cfg,
        "optimization": optim_cfg,
        "model": model_cfg,
    }

    # Also expose the most common dotted aliases for robustness across AG versions.
    if epochs is not None:
        hp["optimization.max_epochs"] = int(epochs)
        hp["optim.max_epochs"] = int(epochs)
    if learning_rate is not None:
        lr_val = float(learning_rate)
        hp["optimization.learning_rate"] = lr_val
        hp["optimization.lr"] = lr_val
        hp["optim.learning_rate"] = lr_val
        hp["optim.lr"] = lr_val
    if batch_size is not None:
        bs_val = int(batch_size)
        hp["optimization.per_device_train_batch_size"] = bs_val
        hp["optimization.batch_size"] = bs_val
        hp["optim.per_device_train_batch_size"] = bs_val
        hp["optim.batch_size"] = bs_val
        hp["env.per_gpu_batch_size"] = bs_val
    if resolved_num_workers is not None:
        hp["env.num_workers"] = int(resolved_num_workers)
    if resolved_num_workers_eval is not None:
        hp["env.num_workers_evaluation"] = int(resolved_num_workers_eval)
    if backbone_image:
        hp["model.timm_image.checkpoint_name"] = str(backbone_image)
    if backbone_text:
        hp["model.hf_text.checkpoint_name"] = str(backbone_text)

    # Merge user-provided hyperparameters (inline JSON or path) last so they win.
    if isinstance(hyperparameters, dict):
        user_hp = hyperparameters
    else:
        user_hp = load_user_hparams(hyperparameters)
    hp = deep_update(hp, user_hp)
    hp = _prune_empty(hp)

    fit_cfg = {}
    if time_limit is not None:
        fit_cfg["time_limit"] = time_limit
    if random_seed is not None:
        fit_cfg["seed"] = int(random_seed)
    if preset:
        fit_cfg["presets"] = preset

    config = {
        "fit": fit_cfg,
        "hyperparameters": hp,
    }
    if threshold is not None:
        config["threshold"] = float(threshold)

    return config


def run_autogluon_experiment(
    train_dataset: pd.DataFrame,
    test_dataset: Optional[pd.DataFrame],
    target_column: str,
    image_columns: Optional[List[str]],
    ag_config: dict,
):
    """
    Launch an AutoGluon MultiModal training run using the config from
    autogluon_hyperparameters(). Returns (predictor, context dict) so callers
    can evaluate downstream with the chosen threshold.
    """
    if ag_config is None:
        raise ValueError("ag_config is required to launch AutoGluon training.")

    hyperparameters = ag_config.get("hyperparameters") or {}
    fit_cfg = dict(ag_config.get("fit") or {})
    threshold = ag_config.get("threshold")

    if "split" not in train_dataset.columns:
        raise ValueError("train_dataset must contain a 'split' column. Did you call split_dataset?")

    df_train = train_dataset[train_dataset["split"] == "train"].copy()
    df_val = train_dataset[train_dataset["split"].isin(["val", "validation"])].copy()
    df_test_internal = train_dataset[train_dataset["split"] == "test"].copy()

    predictor = MultiModalPredictor(label=target_column, path=None)
    column_types = {c: "image_path" for c in (image_columns or [])}

    fit_kwargs = {
        "train_data": df_train,
        "hyperparameters": hyperparameters,
    }
    fit_kwargs.update(fit_cfg)
    if not df_val.empty:
        fit_kwargs.setdefault("tuning_data", df_val)
    if column_types:
        fit_kwargs.setdefault("column_types", column_types)

    logger.info(
        "Fitting AutoGluon with %d train / %d val rows (internal test rows: %d, external test provided: %s)",
        len(df_train),
        len(df_val),
        len(df_test_internal),
        (test_dataset is not None and not test_dataset.empty),
    )
    predictor.fit(**fit_kwargs)

    return predictor, {
        "train": df_train,
        "val": df_val,
        "test_internal": df_test_internal,
        "test_external": test_dataset,
        "threshold": threshold,
    }
