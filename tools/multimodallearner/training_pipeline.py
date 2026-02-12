from __future__ import annotations

import contextlib
import copy
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
_MISSING = object()
_HP_ALIAS_KEYS = [
    "env.num_gpus",
    "env.num_workers",
    "env.num_workers_inference",
    "env.per_gpu_batch_size",
    "optimization.max_epochs",
    "optimization.epochs",
    "optimization.learning_rate",
    "optimization.lr",
    "optimization.batch_size",
    "optimization.per_device_train_batch_size",
    "optimization.train_batch_size",
    "optim.max_epochs",
    "optim.epochs",
    "optim.learning_rate",
    "optim.lr",
    "optim.batch_size",
    "optim.per_device_train_batch_size",
    "model.names",
    "model.hf_text.checkpoint_name",
    "model.timm_image.checkpoint_name",
]


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
    default_value: Optional[int] = None,
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
    if default_value is not None:
        logger.info("Using default %s num_workers=%d (heuristic).", label, int(default_value))
        return int(default_value)
    return None


def _resolve_num_gpus(env_keys: List[str]) -> int:
    env_val = _get_env_int(env_keys)
    if env_val is not None:
        resolved = max(0, int(env_val))
        logger.info("Using env-configured num_gpus=%d.", resolved)
        return resolved
    if not torch.cuda.is_available():
        logger.info("CUDA not available; setting num_gpus=0.")
        return 0
    detected = max(0, int(torch.cuda.device_count()))
    logger.info("Auto-detected GPU count=%d; setting num_gpus=%d.", detected, detected)
    return detected


def _requested_num_gpus(hyperparameters: Dict) -> Optional[int]:
    if not isinstance(hyperparameters, dict):
        return None

    env_cfg = hyperparameters.get("env")
    nested_val = env_cfg.get("num_gpus") if isinstance(env_cfg, dict) else None
    dotted_val = hyperparameters.get("env.num_gpus")
    for val in (nested_val, dotted_val):
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-integer num_gpus value: %s", val)
    return None


def _with_single_gpu(hyperparameters: Dict) -> Dict:
    hp = copy.deepcopy(hyperparameters or {})
    env_cfg = hp.get("env")
    if not isinstance(env_cfg, dict):
        env_cfg = {}
    env_cfg["num_gpus"] = 1
    hp["env"] = env_cfg
    hp["env.num_gpus"] = 1
    return hp


def _enforce_cpu_gpu_safety(hyperparameters: Dict) -> Dict:
    """
    Ensure GPU settings are valid for the current runtime.
    In CPU-only environments, always force num_gpus=0 even if overridden.
    """
    hp = copy.deepcopy(hyperparameters or {})
    env_cfg = hp.get("env")
    if not isinstance(env_cfg, dict):
        env_cfg = {}

    if not torch.cuda.is_available():
        env_cfg["num_gpus"] = 0
        hp["env"] = env_cfg
        hp["env.num_gpus"] = 0
        logger.warning("CUDA is unavailable; forcing num_gpus=0 despite overrides.")
        return hp

    hp["env"] = env_cfg
    return hp


def _looks_like_nccl_dist_init_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "nccl" not in text:
        return False
    markers = (
        "distbackenderror",
        "processraisedexception",
        "init_process_group",
        "operation not supported",
        "unhandled cuda error",
        "cuda failure",
    )
    return any(marker in text for marker in markers)

# ---------------------- small utilities ----------------------


def load_user_hparams(hp_arg: Optional[str]) -> dict:
    """Parse --hyperparameters (inline JSON/YAML or a JSON/YAML file path)."""
    if not hp_arg:
        return {}
    try:
        if isinstance(hp_arg, dict):
            return copy.deepcopy(hp_arg)

        raw = str(hp_arg).strip()
        if not raw:
            return {}

        def _parse_payload(payload: str):
            parsed = None
            json_err = None
            yaml_err = None
            type_err = None

            try:
                parsed = json.loads(payload)
            except Exception as exc:
                json_err = exc

            if parsed is None:
                try:
                    import yaml  # Lazy import; YAML is optional at runtime.
                    parsed = yaml.safe_load(payload)
                except Exception as exc:
                    yaml_err = exc

            if parsed is None:
                return None, json_err, yaml_err, type_err
            if not isinstance(parsed, dict):
                type_err = TypeError(f"expected dict, got {type(parsed).__name__}")
                return None, json_err, yaml_err, type_err
            return parsed, json_err, yaml_err, type_err

        parsed, json_err, yaml_err, type_err = _parse_payload(raw)
        if parsed is not None:
            return parsed

        file_json_err = None
        file_yaml_err = None
        file_type_err = None
        if os.path.exists(raw):
            try:
                with open(raw, "r", encoding="utf-8") as f:
                    payload = f.read()
                parsed, file_json_err, file_yaml_err, file_type_err = _parse_payload(payload)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                logger.warning("Could not read --hyperparameters file '%s': %s", raw, exc)
                return {}

        logger.warning(
            (
                "Could not parse --hyperparameters as inline JSON/YAML or JSON/YAML file; ignoring. "
                "inline_json_error=%s inline_yaml_error=%s inline_type_error=%s "
                "file_json_error=%s file_yaml_error=%s file_type_error=%s"
            ),
            json_err,
            yaml_err,
            type_err,
            file_json_err,
            file_yaml_err,
            file_type_err,
        )
        return {}
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


def _set_nested_key(dst: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = dst
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


def _get_nested_key(src: dict, dotted_key: str, default=_MISSING):
    cur = src
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _apply_dotted_overrides_to_nested(hp: dict) -> dict:
    """
    Mirror dotted keys into nested dicts so user overrides stay consistent.
    """
    normalized = copy.deepcopy(hp or {})
    for key, value in list((hp or {}).items()):
        if isinstance(key, str) and "." in key:
            _set_nested_key(normalized, key, copy.deepcopy(value))
    return normalized


def _synchronize_hparam_aliases(hp: dict) -> dict:
    """
    Keep nested and dotted aliases consistent to avoid conflicting overrides.
    Nested values take precedence when both exist.
    """
    synced = copy.deepcopy(hp or {})
    candidate_keys = set(_HP_ALIAS_KEYS)
    candidate_keys.update(
        key for key in synced.keys()
        if isinstance(key, str) and "." in key
    )

    for dotted_key in sorted(candidate_keys):
        nested_value = _get_nested_key(synced, dotted_key, default=_MISSING)
        has_dotted = dotted_key in synced
        if nested_value is not _MISSING:
            synced[dotted_key] = copy.deepcopy(nested_value)
            continue
        if has_dotted:
            _set_nested_key(synced, dotted_key, copy.deepcopy(synced[dotted_key]))
    return synced


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
        user_hp = _apply_dotted_overrides_to_nested(user_hp)
    else:
        user_hp = {}

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

    if user_hp:
        # Merge user overrides last so explicit custom keys always win.
        hp = deep_update(hp, user_hp)

    if ag_version:
        logger.info(f"Detected AutoGluon version: {ag_version}; applied robust hyperparameter mappings.")

    return _synchronize_hparam_aliases(hp)


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
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, dict]]:
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
    _, raw_metrics, roc_curves = evaluate_all_transparency(
        predictor=predictor,
        train_df=df_train,
        val_df=df_val,
        test_df=df_test_effective,
        target_col=label_col,
        problem_type=problem_type,
        threshold=threshold_test,
    )

    if df_test_external is not None and df_test_external is not df_test and len(df_test_external):
        ext_metrics, ext_curve = compute_metrics_for_split(
            predictor,
            df_test_external,
            label_col,
            problem_type,
            threshold=threshold_test,
            return_curve=True,
        )
        raw_metrics["Test (external)"] = ext_metrics
        if ext_curve:
            roc_curves["Test (external)"] = ext_curve
        ag_by_split["Test (external)"] = ag_evaluate_safely(predictor, df_test_external, metrics=metrics_req)

    return raw_metrics, ag_by_split, roc_curves


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
    default_workers = None
    if shm_bytes is None or shm_bytes >= _LOW_SHM_BYTES:
        cpu_count = os.cpu_count() or 1
        default_workers = max(1, min(8, cpu_count // 2))
    resolved_num_workers = _resolve_num_workers(
        num_workers,
        ["AG_MM_NUM_WORKERS", "AG_NUM_WORKERS", "AUTOMM_NUM_WORKERS"],
        "training",
        shm_bytes,
        default_value=default_workers,
    )
    resolved_num_workers_inference = _resolve_num_workers(
        num_workers_evaluation,
        [
            "AG_MM_NUM_WORKERS_INFERENCE",
            "AG_MM_NUM_WORKERS_EVAL",
            "AG_MM_NUM_WORKERS_EVALUATION",
            "AUTOMM_NUM_WORKERS_EVAL",
        ],
        "inference",
        shm_bytes,
        default_value=default_workers,
    )
    if resolved_num_workers_inference is None and resolved_num_workers is not None:
        resolved_num_workers_inference = resolved_num_workers
    resolved_num_gpus = _resolve_num_gpus(
        ["AG_MM_NUM_GPUS", "AG_NUM_GPUS", "AUTOMM_NUM_GPUS", "NUM_GPUS"],
    )
    if resolved_num_workers is not None:
        env_cfg["num_workers"] = int(resolved_num_workers)
    if resolved_num_workers_inference is not None:
        key = "num_workers_inference"
        env_cfg[key] = int(resolved_num_workers_inference)
    env_cfg["num_gpus"] = int(resolved_num_gpus)

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
    if resolved_num_workers_inference is not None:
        hp[f"env.{key}"] = int(resolved_num_workers_inference)
    hp["env.num_gpus"] = int(resolved_num_gpus)
    if backbone_image:
        hp["model.timm_image.checkpoint_name"] = str(backbone_image)
    if backbone_text:
        hp["model.hf_text.checkpoint_name"] = str(backbone_text)

    # Merge user-provided hyperparameters (inline JSON/YAML or file path) last so they win.
    if isinstance(hyperparameters, dict):
        user_hp = hyperparameters
    else:
        user_hp = load_user_hparams(hyperparameters)
    hp = deep_update(hp, _apply_dotted_overrides_to_nested(user_hp))
    hp = _synchronize_hparam_aliases(hp)
    hp = _enforce_cpu_gpu_safety(hp)
    hp = _synchronize_hparam_aliases(hp)
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
    requested_num_gpus = _requested_num_gpus(hyperparameters)
    try:
        predictor.fit(**fit_kwargs)
    except Exception as exc:
        if not _looks_like_nccl_dist_init_error(exc):
            raise
        if requested_num_gpus == 1:
            raise
        if torch.cuda.device_count() < 2:
            raise

        logger.warning(
            "Detected NCCL distributed initialization failure; retrying with single GPU (env.num_gpus=1). Error: %s",
            type(exc).__name__,
        )
        retry_hyperparameters = _with_single_gpu(hyperparameters)
        fit_kwargs["hyperparameters"] = retry_hyperparameters
        ag_config["hyperparameters"] = retry_hyperparameters
        predictor = MultiModalPredictor(label=target_column, path=None)
        predictor.fit(**fit_kwargs)

    return predictor, {
        "train": df_train,
        "val": df_val,
        "test_internal": df_test_internal,
        "test_external": test_dataset,
        "threshold": threshold,
    }
