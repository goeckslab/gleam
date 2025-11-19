from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import zipfile
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import StratifiedKFold, KFold

from split_logic import (
    load_and_split,
    path_expander_any,
)
from plot_logic import (
    infer_problem_type,
    build_summary_html,
    build_test_html_and_plots,
    build_feature_html,
    assemble_full_html_report,
    build_train_html_and_plots,
)
from metrics_logic import evaluate_all_transparency  # kept for type hints; main eval in training_pipeline

# Transparency helpers (report_utils.py)
from report_utils import (
    collect_run_context,
    build_class_balance_html,
    build_leaderboard_html,
    build_ignored_features_html,
    build_presets_hparams_html,
    build_warnings_html,
    build_reproducibility_html,
    build_model_performance_summary_table,
    get_model_architecture,
)

# NEW: training / evaluation core
from training_pipeline import (
    build_mm_hparams,
    train_predictor,
    evaluate_predictor_all_splits,
    fit_summary_safely,
)

# ------------- Logging -------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Quiet noisy libs
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def enable_tensor_cores_if_available():
    try:
        torch.set_float32_matmul_precision("high")
        logger.info("Enabled torch float32 matmul precision = 'high' (Tensor Cores).")
    except Exception:
        pass


def ensure_local_tmp():
    for d in ("/dev/shm", "/tmp"):
        try:
            if os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK):
                os.environ.setdefault("TMPDIR", d)
                tempfile.tempdir = d
                logger.info(f"Using local TMPDIR: {d}")
                return d
        except Exception:
            pass
    logger.info("Using default TMPDIR")
    return None


def normalize_image_args(args):
    """Normalize image arguments on the parsed args object.

    - `--image_columns` accepts comma-separated string or repeated args and is stored as a list.
    - `--images_zips` becomes a clean list (commas or repeated values allowed).
    - `--image_folders` is filtered to existing directories only.
    The function mutates `args` in-place.
    """
    # Normalize image columns to a list or None
    img_cols = args.image_columns
    if isinstance(img_cols, str):
        img_cols = [c.strip() for c in img_cols.split(",") if c.strip()]
    elif img_cols is not None:
        img_cols = [str(c).strip() for c in img_cols if str(c).strip()]
    args.image_columns = img_cols or None

    # Normalize image zips into a list
    zips = args.images_zips
    if isinstance(zips, str):
        zips = [z.strip() for z in zips.split(",") if z.strip()]
    elif zips is not None:
        zips = [str(z).strip() for z in zips if str(z).strip()]
    args.images_zips = zips or []

    # image_folders: keep only paths that exist and are directories
    args.image_folders = [d for d in (args.image_folders or []) if d and os.path.isdir(d)]


def parse_args(argv=None):
    """Build and validate CLI arguments. Returns argparse.Namespace.

    Kept separate to make `main()` easier to read and test.
    """
    parser = argparse.ArgumentParser(description="Train & report an AutoGluon model")
    parser.add_argument("--input_csv_train", dest="train_csv", required=True)
    parser.add_argument("--input_csv_test", dest="test_csv", default=None)
    parser.add_argument("--target_column", dest="label_column", required=True)
    parser.add_argument("--output_csv", dest="output_csv", required=True)
    parser.add_argument("--output_json", dest="output_json", default="results.json")
    parser.add_argument("--output_html", dest="output_html", default="report.html")

    # Images (lists + legacy)
    parser.add_argument("--image_columns", dest="image_columns", nargs="+", default=None)
    parser.add_argument("--images_zips", dest="images_zips", nargs="*", default=None)
    parser.add_argument("--image_folders", dest="image_folders", nargs="*", default=None)
    parser.add_argument("--text_columns", dest="text_columns", nargs="+", default=None,
                        help="Optional list of columns that contain free-form text inputs")
    parser.add_argument("--use_text", dest="use_text", type=str, default="true",
                        help="true/false: whether to enable text modalities (default: true)")

    # How to handle missing images: if true -> remove rows with missing images; if false -> inject placeholder image path
    parser.add_argument("--missing_image_strategy", dest="missing_image_strategy", default="false",
                        help="true/false: if true remove rows with missing image paths; if false, generate placeholder image to fill missing entries")

    # Threshold only for Test
    parser.add_argument("--threshold", dest="threshold", type=float, default=None)

    parser.add_argument("--time_limit", dest="time_limit", type=int, default=None)
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=42)

    # New training knobs
    parser.add_argument("--cross_validation", dest="cross_validation", type=str, default="false",
                        help="Activate cross-validation: true or false")
    parser.add_argument("--num_folds", dest="num_folds", type=int, default=5,
                        help="Number of folds for cross-validation (integer)")
    parser.add_argument("--epochs", dest="epochs", type=int, default=None,
                        help="Number of training epochs (optional)")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=None,
                        help="Learning rate for training (optional)")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=None,
                        help="Batch size for training (optional)")
    # Backbone selection per modality
    parser.add_argument("--backbone_image", dest="backbone_image", type=str, default="swin_base_patch4_window7_224",
                        help="Image backbone / timm checkpoint name for AutoMM (default: swin_base_patch4_window7_224)")
    parser.add_argument("--backbone_text", dest="backbone_text", type=str, default="microsoft/deberta-v3-base",
                        help="Text backbone / HF checkpoint for AutoMM (default: microsoft/deberta-v3-base)")
    parser.add_argument("--backbone_tabular", dest="backbone_tabular", type=str, default="ft_transformer",
                        help="Structured backbone selection mapped to mm_hparams.model.tabular.backbone (default: ft_transformer)")
    parser.add_argument("--backbone_fusion", dest="backbone_fusion", type=str, default="fusion_transformer",
                        help="Fusion backbone that combines modalities before the prediction head (fusion_mlp or fusion_transformer)")

    # Split knobs
    parser.add_argument("--validation_size", type=float, default=0.125)
    parser.add_argument("--split_probabilities", type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        metavar=("train", "val", "test"))
    parser.add_argument("--val_size_with_test", type=float, default=0.2)

    # Cheat-sheet knobs
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument("--preset", dest="preset", choices=["medium_quality", "high_quality", "best_quality"], default=None,
                        help="Single preset: medium_quality, high_quality, or best_quality")
    parser.add_argument("--eval_metric", default="roc_auc",
                        help="Evaluation metric to use for training/evaluation (default: roc_auc)")
    parser.add_argument("--hyperparameters", default=None)

    args = parser.parse_args(argv)

    # Normalize legacy/plural image arguments
    normalize_image_args(args)

    # Normalize boolean-like CLI values
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        try:
            return str(v).strip().lower() in ("true", "1", "yes", "y")
        except Exception:
            return False

    args.cross_validation = _str2bool(args.cross_validation)
    args.missing_image_strategy = _str2bool(args.missing_image_strategy)
    args.use_text = _str2bool(args.use_text)
    # If user provided single --preset, prefer that and expose as args.presets for downstream compatibility
    if getattr(args, "preset", None):
        args.presets = [args.preset]
    # If no presets were provided at all, default to high_quality
    if not getattr(args, "presets", None):
        args.presets = ["high_quality"]

    # Basic validation
    if not (0.0 <= args.validation_size <= 1.0):
        parser.error("--validation_size must be in [0, 1]")
    if len(args.split_probabilities) != 3 or abs(sum(args.split_probabilities) - 1.0) > 1e-6:
        parser.error("--split_probabilities must be three numbers summing to 1.0")
    if not (0.0 < args.val_size_with_test < 1.0):
        parser.error("--val_size_with_test must be in (0, 1)")

    if args.cross_validation and (args.num_folds is None or args.num_folds < 2):
        parser.error("--num_folds must be an integer >= 2 when --cross_validation is true")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be a positive integer if specified")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        parser.error("--learning_rate must be > 0 if specified")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer if specified")

    return args


def verify_outputs(paths):
    ok = True
    for p, desc in paths:
        if os.path.exists(p):
            size = os.path.getsize(p)
            logger.info(f"✓ Output {desc}: {p} ({size:,} bytes)")
            os.chmod(p, 0o644)
        else:
            logger.error(f"✗ Output {desc} MISSING: {p}")
            ok = False
    if not ok:
        logger.error("Some outputs are missing!")
        sys.exit(1)


# Fallback helper if needed later
SPLIT_COLUMN_NAME = "split"

def create_stratified_random_split(
    df: pd.DataFrame,
    split_column: str,
    split_probabilities: List[float],
    random_state: int,
    label_column: str,
) -> pd.DataFrame:
    p_train, p_val, p_test = split_probabilities
    df = df.copy()
    rng = np.random.RandomState(int(random_state))
    df[split_column] = 0
    for _cls, grp in df.groupby(label_column, dropna=False):
        idx = grp.sample(frac=1.0, random_state=rng.randint(0, 10**9)).index
        n = len(idx)
        n_train = int(round(n * p_train))
        n_val = int(round(n * p_val))
        n_train = max(0, min(n, n_train))
        n_val = max(0, min(n - n_train, n_val))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        df.loc[val_idx, split_column] = 1
        df.loc[test_idx, split_column] = 2
    return df


def main():
    args = parse_args()

    # Debug
    logger.info("=== Galaxy Tool Debug Info ===")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Parsed arguments: {vars(args)}")
    logger.info(f"Input train CSV exists: {os.path.exists(args.train_csv)}")
    if args.images_zips:
        logger.info(f"Image ZIPs count: {len(args.images_zips)} | all exist? {[os.path.exists(z) for z in args.images_zips]}")
    logger.info("=== End Debug Info ===")

    # Perf & reproducibility
    set_seeds(args.random_seed)
    ensure_local_tmp()
    enable_tensor_cores_if_available()

    # Build base folders (extract zips into a single shared directory first)
    base_folders: List[str] = list(args.image_folders or [])
    extracted_images_dir = None
    if args.images_zips:
        extracted_images_dir = tempfile.mkdtemp(prefix="images_zip_")
        for z in args.images_zips:
            if not os.path.isfile(z):
                logger.warning(f"Image ZIP '{z}' does not exist or is not a file; skipping.")
                continue
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(extracted_images_dir)
            logger.info(f"Extracted '{z}' into shared directory {extracted_images_dir}")
        if extracted_images_dir:
            base_folders.insert(0, extracted_images_dir)
    base_folders.append(os.getcwd())

    # Load + split
    try:
        df_train, df_val, df_test, df_train_full, label_col, image_cols, text_cols = load_and_split(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            label_column=args.label_column,
            image_columns=args.image_columns,
            text_columns=args.text_columns,
            random_seed=args.random_seed,
            validation_size=args.validation_size,
            split_probabilities=args.split_probabilities,
            val_size_with_test=args.val_size_with_test,
        )
        args.label_column = label_col
        args.image_columns = image_cols
        args.text_columns = text_cols
    except Exception as e:
        logger.error(f"Failed to read/split input CSVs: {e}")
        sys.exit(1)

    # Verify cols exist
    for df_, name in [(df_train_full, "train"), (df_test, "test")]:
        if args.label_column not in df_.columns:
            logger.error(f"Missing target column '{args.label_column}' in {name} CSV")
            sys.exit(1)
        if args.image_columns:
            for c in args.image_columns:
                if c not in df_.columns:
                    logger.error(f"Missing image column '{c}' in {name} CSV")
                    sys.exit(1)
        if args.text_columns:
            for c in args.text_columns:
                if c not in df_.columns:
                    logger.error(f"Missing text column '{c}' in {name} CSV")
                    sys.exit(1)

    # Expand image paths to absolute locations for every dataframe copy we use downstream
    if args.image_columns:
        for df_ in (df_train, df_val, df_test, df_train_full):
            if df_ is None:
                continue
            for c in args.image_columns:
                df_[c] = df_[c].astype(str).apply(lambda p: path_expander_any(p, base_folders))

    # Handle missing images according to missing_image_strategy
    if args.image_columns:
        # helper to detect missing/invalid paths
        def _is_missing_path(p: str) -> bool:
            try:
                if p is None:
                    return True
                ps = str(p).strip()
                if ps == "" or ps.lower() in ("nan", "none"):
                    return True
                return not os.path.exists(ps)
            except Exception:
                return True

        placeholder_path = None
        # If strategy is False -> generate placeholder image and fill missing entries
        if not args.missing_image_strategy:
            try:
                # create a single placeholder image to reuse
                import uuid
                try:
                    from PIL import Image
                    pillow_ok = True
                except Exception:
                    pillow_ok = False
                placeholder_dir = tempfile.mkdtemp(prefix="placeholder_images_")
                placeholder_path = os.path.join(placeholder_dir, f"placeholder_{uuid.uuid4().hex}.png")
                if pillow_ok:
                    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
                    img.save(placeholder_path, format="PNG")
                else:
                    # Fallback to matplotlib if Pillow not available
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as _np
                        arr = (_np.ones((64, 64, 3), dtype=_np.uint8) * 200)
                        plt.imsave(placeholder_path, arr)
                        plt.close("all")
                    except Exception:
                        placeholder_path = None
                if placeholder_path:
                    logger.info(f"Generated placeholder image at {placeholder_path} for missing image entries")
            except Exception as e:
                logger.warning(f"Could not create placeholder image: {e}")

        # For each dataframe, either drop rows with missing images or fill with placeholder
        for df_name, df_ in (("train", df_train), ("val", df_val), ("test", df_test)):
            if df_ is None or len(df_) == 0:
                continue
            missing_any = None
            for c in args.image_columns:
                mask = df_[c].fillna("").astype(str).apply(_is_missing_path)
                if missing_any is None:
                    missing_any = mask
                else:
                    missing_any = missing_any | mask

            if missing_any is None:
                continue

            n_missing = int(missing_any.sum())
            if n_missing == 0:
                continue

            if args.missing_image_strategy:
                # Drop rows with any missing images
                logger.info(f"Dropping {n_missing} rows from {df_name} due to missing image files (per --missing_image_strategy=true)")
                # mutate the respective df reference
                if df_name == "train":
                    df_train = df_train.loc[~missing_any].reset_index(drop=True)
                elif df_name == "val":
                    df_val = df_val.loc[~missing_any].reset_index(drop=True)
                else:
                    df_test = df_test.loc[~missing_any].reset_index(drop=True)
            else:
                # Fill missing image entries with placeholder path so AutoMM doesn't error
                if not placeholder_path:
                    logger.warning("No placeholder image available; missing image entries will remain as-is and may cause errors")
                else:
                    logger.info(f"Filling {n_missing} missing image entries in {df_name} with placeholder image")
                    for c in args.image_columns:
                        try:
                            df_.loc[missing_any, c] = placeholder_path
                        except Exception:
                            # older pandas versions may require assignment differently
                            df_.loc[missing_any.values, c] = placeholder_path
                    # write back updated reference
                    if df_name == "train":
                        df_train = df_.reset_index(drop=True)
                    elif df_name == "val":
                        df_val = df_.reset_index(drop=True)
                    else:
                        df_test = df_.reset_index(drop=True)

    # If cross-validation is requested, run k-fold training over df_train_full (train+val)
    if args.cross_validation:
        logger.info(f"Running cross-validation with {args.num_folds} folds")
        df_full = df_train_full.reset_index(drop=True)
        y = df_full[args.label_column]
        # Choose stratified split for classification-like targets when possible
        try:
            use_stratified = y.dtype == object or y.nunique() <= 20
        except Exception:
            use_stratified = False

        kf = StratifiedKFold(n_splits=int(args.num_folds), shuffle=True, random_state=int(args.random_seed)) if use_stratified else KFold(n_splits=int(args.num_folds), shuffle=True, random_state=int(args.random_seed))

        raw_folds = []
        ag_folds = []
        last_predictor = None
        fold_idx = 0
        for train_idx, val_idx in kf.split(df_full, y if use_stratified else None):
            fold_idx += 1
            logger.info(f"CV fold {fold_idx}/{args.num_folds}")
            df_tr = df_full.iloc[train_idx].copy()
            df_va = df_full.iloc[val_idx].copy()

            # Expand image paths for these fold-specific frames
            if args.image_columns:
                for c in args.image_columns:
                    df_tr[c] = df_tr[c].astype(str).apply(lambda p: path_expander_any(p, base_folders))
                    df_va[c] = df_va[c].astype(str).apply(lambda p: path_expander_any(p, base_folders))

            try:
                mm_hparams_fold = build_mm_hparams(args, df_tr, args.image_columns)
                predictor_fold = train_predictor(args, df_tr, df_va, args.image_columns, mm_hparams_fold)
                last_predictor = predictor_fold
                raw_metrics_fold, ag_by_split_fold = evaluate_predictor_all_splits(
                    predictor=predictor_fold,
                    df_train=df_tr,
                    df_val=df_va,
                    df_test=df_test,
                    label_col=args.label_column,
                    problem_type=infer_problem_type(predictor_fold, df_tr, args.label_column),
                    eval_metric=args.eval_metric,
                    threshold_test=args.threshold,
                )
                # capture predictor path if available
                pred_path_fold = getattr(predictor_fold, "path", None)
                raw_folds.append(raw_metrics_fold)
                ag_folds.append(ag_by_split_fold)
                # store fold-level info (metrics + predictor path)
                if 'folds_info' not in locals():
                    folds_info = []
                folds_info.append({
                    "fold": int(fold_idx),
                    "predictor_path": pred_path_fold,
                    "raw_metrics": raw_metrics_fold,
                    "ag_eval": ag_by_split_fold,
                })
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")

        # Aggregate folds (mean of numeric entries)
        def _aggregate(list_of_metrics):
            # list_of_metrics: list of raw_metrics dicts
            agg_mean = {}
            agg_std = {}
            for split in ("Train", "Validation", "Test"):
                # collect keys
                keys = set()
                for m in list_of_metrics:
                    if split in m:
                        keys.update(m[split].keys())
                if not keys:
                    continue
                agg_mean[split] = {}
                agg_std[split] = {}
                for k in keys:
                    vals = [m[split][k] for m in list_of_metrics if split in m and k in m[split]]
                    # try numeric aggregation
                    numeric_vals = []
                    for v in vals:
                        try:
                            numeric_vals.append(float(v))
                        except Exception:
                            pass
                    if numeric_vals:
                        mean_v = float(np.mean(numeric_vals))
                        std_v = float(np.std(numeric_vals, ddof=0))
                        agg_mean[split][k] = mean_v
                        agg_std[split][k] = std_v
                    else:
                        # fallback: keep last value as-is
                        agg_mean[split][k] = vals[-1] if vals else None
                        agg_std[split][k] = None
            return agg_mean, agg_std

        raw_metrics, raw_metrics_std = _aggregate(raw_folds)

        # Aggregate AutoGluon evals similarly (mean + std)
        ag_by_split = {"Train": {}, "Validation": {}, "Test": {}}
        ag_by_split_std = {"Train": {}, "Validation": {}, "Test": {}}
        for split in ("Train", "Validation", "Test"):
            keys = set()
            for m in ag_folds:
                if split in m:
                    keys.update(m[split].keys())
            for k in keys:
                vals = [m[split][k] for m in ag_folds if split in m and k in m[split]]
                numeric_vals = []
                for v in vals:
                    try:
                        numeric_vals.append(float(v))
                    except Exception:
                        pass
                if numeric_vals:
                    ag_by_split[split][k] = float(np.mean(numeric_vals))
                    ag_by_split_std[split][k] = float(np.std(numeric_vals, ddof=0))
                else:
                    ag_by_split[split][k] = vals[-1] if vals else None
                    ag_by_split_std[split][k] = None

        predictor = last_predictor
        if predictor is None:
            logger.error("All CV folds failed. Exiting.")
            sys.exit(1)

        # Persist per-fold metrics for inclusion in outputs
        try:
            folds_payload = {
                "raw_folds": raw_folds,
                "ag_folds": ag_folds,
                "folds_info": folds_info if 'folds_info' in locals() else None,
                "summary_mean": raw_metrics,
                "summary_std": raw_metrics_std,
                "ag_summary_mean": ag_by_split,
                "ag_summary_std": ag_by_split_std,
            }
            with open("folds_metrics.json", "w") as ff:
                json.dump(folds_payload, ff, indent=2, default=str)
            logger.info("Wrote per-fold metrics → folds_metrics.json")
        except Exception as e:
            logger.warning(f"Could not write per-fold metrics file: {e}")

    else:
        # Build hparams & train
        mm_hparams = build_mm_hparams(args, df_train, args.image_columns)
        predictor = train_predictor(args, df_train, df_val, args.image_columns, mm_hparams)

        # Authoritative metrics from final predictor + transparent suite
        raw_metrics, ag_by_split = evaluate_predictor_all_splits(
            predictor=predictor,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            label_col=args.label_column,
            problem_type=infer_problem_type(predictor, df_train_full, args.label_column),
            eval_metric=args.eval_metric,
            threshold_test=args.threshold,
        )

    # Fallback if val/test got empty
    if (len(df_val) == 0) or (len(df_test) == 0):
        sys.stderr.write(
            "WARNING: Empty validation or test set after fixed split; "
            "falling back to stratified random split using --split_probabilities.\n"
        )
        try:
            df_full = pd.read_csv(args.train_csv)
        except Exception as e:
            logger.error(f"Could not reload full train CSV for fallback split: {e}")
            sys.exit(1)
        if args.label_column not in df_full.columns:
            logger.error(f"Fallback split failed: label column '{args.label_column}' not found in train CSV.")
            sys.exit(1)
        df_split = create_stratified_random_split(
            df=df_full.copy(),
            split_column=SPLIT_COLUMN_NAME,
            split_probabilities=list(args.split_probabilities),
            random_state=args.random_seed,
            label_column=args.label_column,
        )
        df_train = df_split[df_split[SPLIT_COLUMN_NAME] == 0].copy()
        df_val   = df_split[df_split[SPLIT_COLUMN_NAME] == 1].copy()
        df_test  = df_split[df_split[SPLIT_COLUMN_NAME] == 2].copy()
        df_train_full = df_split[df_split[SPLIT_COLUMN_NAME].isin([0, 1])].copy()
        logger.info(f"(Fallback) Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    logger.info(f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    # Capture warnings
    caught_warnings: List[str] = []
    def _warn_recorder(message, category, filename, lineno, file=None, line=None):
        try:
            caught_warnings.append(f"{category.__name__}: {message}")
        except Exception:
            pass
    warnings.showwarning = _warn_recorder
    warnings.filterwarnings("default")

    # Save predictor path
    try:
        pred_path = getattr(predictor, "path", None)
        if pred_path:
            with open("predictor_path.txt", "w") as pf:
                pf.write(str(pred_path))
            logger.info(f"Wrote predictor path → predictor_path.txt ({pred_path})")
    except Exception:
        logger.warning("Could not write predictor_path.txt")

    # Problem type
    kind = infer_problem_type(predictor, df_train_full, args.label_column)
    logger.info(f"Inferred problem type: {kind}")

    # Authoritative metrics from final predictor + transparent suite
    raw_metrics, ag_by_split = evaluate_predictor_all_splits(
        predictor=predictor,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        label_col=args.label_column,
        problem_type=kind,
        eval_metric=args.eval_metric,
        threshold_test=args.threshold,
    )

    # Inject AG eval metrics into raw_metrics for visibility
    def _inject_ag(src: dict, dst: dict):
        for k, v in (src or {}).items():
            dst[f"AG_{k}"] = float(v)
    if "Train" in raw_metrics:      _inject_ag(ag_by_split["Train"], raw_metrics["Train"])
    if "Validation" in raw_metrics: _inject_ag(ag_by_split["Validation"], raw_metrics["Validation"])
    if "Test" in raw_metrics:       _inject_ag(ag_by_split["Test"], raw_metrics["Test"])

    # CSV
    all_keys: List[str] = []
    for split in ("Train", "Validation", "Test"):
        if split in raw_metrics:
            for k in raw_metrics[split].keys():
                if k not in all_keys:
                    all_keys.append(k)

    rows = []
    if "Train" in raw_metrics:
        rows.append({"phase": "train", **{k: raw_metrics["Train"].get(k, np.nan) for k in all_keys}})
    if "Validation" in raw_metrics:
        rows.append({"phase": "validation", **{k: raw_metrics["Validation"].get(k, np.nan) for k in all_keys}})
    if "Test" in raw_metrics:
        rows.append({"phase": "test", **{k: raw_metrics["Test"].get(k, np.nan) for k in all_keys}})

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    logger.info(f"Wrote metrics CSV → {args.output_csv}")

    # JSON
    fit_summary_obj: Optional[dict] = fit_summary_safely(predictor)
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "train": raw_metrics.get("Train", {}),
                "val": raw_metrics.get("Validation", {}),
                "test": raw_metrics.get("Test", {}),
                "ag_eval": {
                    "train": ag_by_split.get("Train", {}),
                    "val":   ag_by_split.get("Validation", {}),
                    "test":  ag_by_split.get("Test", {}),
                },
                "fit_summary": fit_summary_obj,
                "problem_type": kind,
                "predictor_path": getattr(predictor, "path", None),
                "threshold": args.threshold,
                "threshold_test": args.threshold,
                "presets": args.presets,
                "eval_metric": args.eval_metric,
                "folds": {
                    "raw_folds": raw_folds if getattr(args, "cross_validation", False) else None,
                    "ag_folds": ag_folds if getattr(args, "cross_validation", False) else None,
                    "folds_info": folds_info if (getattr(args, "cross_validation", False) and 'folds_info' in locals()) else None,
                    "summary_mean": raw_metrics if getattr(args, "cross_validation", False) else None,
                    "summary_std": raw_metrics_std if getattr(args, "cross_validation", False) else None,
                    "ag_summary_mean": ag_by_split if getattr(args, "cross_validation", False) else None,
                    "ag_summary_std": ag_by_split_std if getattr(args, "cross_validation", False) else None,
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Wrote full JSON → {args.output_json}")

    # ---------------- HTML report ----------------
    tmpdir = tempfile.mkdtemp()

    label_col = args.label_column
    image_cols = args.image_columns or []
    img_cols_display = ", ".join(image_cols) if image_cols else "—"

    exclude_cols = set(image_cols) | {label_col}
    text_cols = [
        c for c in df_train_full.columns
        if c not in exclude_cols
        and str(df_train_full[c].dtype) == "object"
        and df_train_full[c].notna().any()
    ]
    text_cols_display = ", ".join(text_cols) if text_cols else "—"
    tabular_cols = [c for c in df_train_full.columns if c not in exclude_cols]
    tabular_count = len(tabular_cols)

    presets_used = " ".join(args.presets) if args.presets else "AutoGluon default"
    time_limit_val = "None" if args.time_limit is None else str(int(args.time_limit))

    # --- Custom: Extract model names, backbones, and training knobs from config.yaml in predictor_path.txt ---
    model_arch_names = None
    image_backbone = None
    text_backbone = None
    structured_backbone = None
    image_modality_present = False
    text_modality_present = False
    structured_modality_present = False
    cfg_epochs = None
    cfg_lr = None
    cfg_batch = None
    config_data = None
    try:
        pred_path_file = "predictor_path.txt"
        if os.path.exists(pred_path_file):
            with open(pred_path_file, "r") as pf:
                pred_path = pf.read().strip()
            config_path = os.path.join(pred_path, "config.yaml")
            if os.path.exists(config_path):
                import yaml
                with open(config_path, "r") as cf:
                    config_data = yaml.safe_load(cf) or {}
                model_section = config_data.get("model", {})
                model_arch_names = model_section.get("names", None)
                if model_arch_names:
                    arch_str = ", ".join(str(m) for m in model_arch_names)
                else:
                    arch_str = get_model_architecture(predictor)
                # Extract image backbone if present
                timm_image = model_section.get("timm_image", {}) or {}
                if isinstance(timm_image, dict) and timm_image:
                    image_modality_present = True
                    image_backbone = (
                        timm_image.get("checkpoint_name")
                        or timm_image.get("name")
                        or image_backbone
                    )
                # Extract text backbone if present
                hf_text_section = model_section.get("hf_text", {}) or {}
                if isinstance(hf_text_section, dict) and hf_text_section:
                    text_modality_present = True
                    text_backbone = (
                        hf_text_section.get("checkpoint_name")
                        or hf_text_section.get("name")
                        or text_backbone
                    )
                # Extract structured backbone if present
                ft_transformer = model_section.get("ft_transformer", {}) or {}
                if isinstance(ft_transformer, dict) and ft_transformer:
                    structured_modality_present = True
                    structured_backbone = ft_transformer.get("embedding_arch", None) or structured_backbone
                if isinstance(structured_backbone, list):
                    structured_backbone = ", ".join(str(x) for x in structured_backbone)
                if not structured_modality_present:
                    for key in ("numerical_mlp", "categorical_mlp", "tabular", "fusion_mlp"):
                        if isinstance(model_section.get(key), dict):
                            structured_modality_present = True
                            break
                # Extract training knobs
                optim_section = config_data.get("optim", {})
                cfg_epochs = optim_section.get("max_epochs", None) or optim_section.get("epochs", None)
                cfg_lr = optim_section.get("lr", None) or optim_section.get("learning_rate", None)
                env_section = config_data.get("env", {})
                cfg_batch = env_section.get("batch_size", None) or env_section.get("per_gpu_batch_size", None)
            else:
                arch_str = get_model_architecture(predictor)
        else:
            arch_str = get_model_architecture(predictor)
    except Exception:
        arch_str = get_model_architecture(predictor)

    if not image_modality_present:
        image_modality_present = bool(image_cols)
    if not structured_modality_present:
        structured_modality_present = tabular_count > 0

    # Determine presets used (prefer single --preset then --presets list)
    if getattr(args, "preset", None):
        presets_used = args.preset
    else:
        presets_used = " ".join(args.presets) if args.presets else "AutoGluon default"

    # Build the extra run rows in the requested order and with renames
    extra_run_rows = []
    extra_run_rows.append(("Model architecture", arch_str))
    # Insert backbones immediately after model architecture if present
    if image_backbone:
        extra_run_rows.append(("Image backbone", image_backbone))
    if text_modality_present:
        extra_run_rows.append(("Text backbone", text_backbone or "—"))
    if structured_backbone:
        extra_run_rows.append(("Structured backbone", structured_backbone))

    # Modalities (renamed)
    extra_run_rows.append(("Modalities", "MultiModalPredictor (images + structured/tabular)"))
    extra_run_rows.append(("Label column", args.label_column))
    if image_modality_present:
        extra_run_rows.append(("Unstructured - Image", img_cols_display))
    if text_cols or text_modality_present:
        extra_run_rows.append(("Unstructured - Text", text_cols_display))
    if structured_modality_present:
        extra_run_rows.append(("Structured - numeric/categorical", str(tabular_count)))
    # Experiment quality (presets)
    extra_run_rows.append(("Experiment quality", presets_used))
    # Model evaluation metric (renamed)
    extra_run_rows.append(("Model Evaluation Metric", args.eval_metric or "AutoGluon default"))
    extra_run_rows.append(("Seed", str(int(args.random_seed))))
    extra_run_rows.append(("time limit(s)", time_limit_val))

    # Epochs / LR / Batch from config.yaml if present, else CLI knobs
    epochs_val = cfg_epochs if cfg_epochs is not None else (args.epochs if args.epochs is not None else "—")
    lr_val = cfg_lr if cfg_lr is not None else (args.learning_rate if args.learning_rate is not None else "—")
    batch_val = cfg_batch if cfg_batch is not None else (args.batch_size if args.batch_size is not None else "—")
    extra_run_rows.append(("Epochs", str(epochs_val)))
    extra_run_rows.append(("Learning Rate", str(lr_val)))
    extra_run_rows.append(("Batch Size", str(batch_val)))

    class_balance_block_html = build_class_balance_html(df_train_full, label_col)

    summary_perf_table_html = build_model_performance_summary_table(
        train_scores=raw_metrics.get("Train", {}),
        val_scores=raw_metrics.get("Validation", {}),
        test_scores=raw_metrics.get("Test", {}),
        include_test=True,
        title=None,
        show_title=False,
    )
    # Get feature importance HTML
    feature_importance_html = None
    try:
        from feature_importance import build_feature_importance_html
        feature_importance_html = build_feature_importance_html(predictor, df_train_full, args.label_column)
    except Exception as e:
        feature_importance_html = f"<p>Error loading feature importance: {e}</p>"

    summary_html = build_summary_html(
        predictor=predictor,
        df_train=df_train_full,
        df_val=df_val,
        df_test=df_test,
        label_column=args.label_column,
        extra_run_rows=extra_run_rows,
        class_balance_html=class_balance_block_html,
        perf_table_html=summary_perf_table_html,
        feature_html=feature_importance_html,
    )

    train_tab_perf_html = build_model_performance_summary_table(
        train_scores=raw_metrics.get("Train", {}),
        val_scores=raw_metrics.get("Validation", {}),
        test_scores=raw_metrics.get("Test", {}),
        include_test=False,
        title=None,
        show_title=False,
    )
    train_html = build_train_html_and_plots(
        predictor=predictor,
        problem_type=kind,
        df_train=df_train,
        label_column=args.label_column,
        tmpdir=tmpdir,
        seed=int(args.random_seed),
        perf_table_html=train_tab_perf_html,
        threshold=None,
    )

    test_html_template, plots = build_test_html_and_plots(
        predictor,
        kind,
        df_test,
        args.label_column,
        tmpdir,
        threshold=args.threshold,
    )

    def _fmt_val(v):
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.6f}"
        return str(v)

    test_scores = raw_metrics.get("Test", {})
    metric_rows = "".join(
        f"<tr><td>{k.replace('_',' ').replace('(TNR)','(TNR)').replace('(Sensitivity/TPR)', '(Sensitivity/TPR)')}</td>"
        f"<td>{_fmt_val(v)}</td></tr>"
        for k, v in test_scores.items()
    )
    test_html_filled = test_html_template.format(metric_rows)

    is_multimodal = isinstance(predictor, MultiModalPredictor)
    if is_multimodal:
        feature_text = (
            "<p>Permutation importance is not supported for MultiModalPredictor in this tool. "
            "For tabular-only runs, this section shows permutation importance.</p>"
        )
    else:
        feature_text = build_feature_html(predictor, df_test, args.label_column, tmpdir, args.random_seed)

    # If CV was used, build a simple per-fold metrics HTML block to include in the report
    folds_html = ""
    if getattr(args, "cross_validation", False):
        try:
            # Build a summary table (mean ± std) for each split
            summary_blocks = []
            try:
                for split in ("Train", "Validation", "Test"):
                    mean_dict = raw_metrics.get(split, {}) if isinstance(raw_metrics, dict) else {}
                    std_dict = raw_metrics_std.get(split, {}) if isinstance(raw_metrics_std, dict) else {}
                    if mean_dict:
                        rows = [f"<h4>{split} summary (mean ± std)</h4>", "<table class=\"fold-summary\">"]
                        for k in sorted(mean_dict.keys()):
                            m = mean_dict.get(k)
                            s = std_dict.get(k) if std_dict is not None else None
                            if s is None:
                                val = f"{m}"
                            else:
                                val = f"{m:.6f} ± {s:.6f}" if isinstance(m, float) and isinstance(s, float) else f"{m} ± {s}"
                            rows.append(f"<tr><td>{k}</td><td>{val}</td></tr>")
                        rows.append("</table>")
                        summary_blocks.append("\n".join(rows))
            except Exception:
                # ignore summary build errors
                pass

            folds_list_html = []
            for i, (rf, af) in enumerate(zip(raw_folds, ag_folds), start=1):
                fold_rows = []
                fold_rows.append(f"<h4>Fold {i}</h4>")
                # include predictor path if available
                try:
                    pinfo = None
                    if 'folds_info' in locals() and len(folds_info) >= i:
                        pinfo = folds_info[i-1].get('predictor_path')
                    if pinfo:
                        fold_rows.append(f"<p>Model path: {pinfo}</p>")
                except Exception:
                    pass
                # Raw metrics
                if rf:
                    for split_name, metrics_dict in rf.items():
                        fold_rows.append(f"<h5>{split_name}</h5>")
                        fold_rows.append("<table class=\"fold-table\">")
                        for k, v in metrics_dict.items():
                            fold_rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
                        fold_rows.append("</table>")
                # AG evals
                if af:
                    fold_rows.append("<h5>AutoGluon eval</h5>")
                    for split_name, metrics_dict in af.items():
                        fold_rows.append(f"<h6>{split_name}</h6>")
                        fold_rows.append("<table class=\"fold-table\">")
                        for k, v in metrics_dict.items():
                            fold_rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
                        fold_rows.append("</table>")
                folds_list_html.append("\n".join(fold_rows))

            folds_html = "<section><h3>Cross-validation fold metrics</h3>" + "<hr>".join(summary_blocks + folds_list_html) + "</section>"
        except Exception:
            folds_html = "<p>Could not build fold metrics HTML.</p>"

    notices: List[str] = []
    notices.append("No presets specified; AutoGluon defaulted to 'medium' (fast prototyping)." if not args.presets else f"Presets used: {presets_used}.")
    if kind in ("binary", "multiclass") and len(df_val) < 10_000:
        notices.append("Decision threshold calibration disabled due to <10,000 validation rows (to avoid overfitting).")
    if args.threshold is not None and kind == "binary":
        notices.append(f"Using decision threshold = {float(args.threshold):.3f} on Test only.")
    if os.environ.get("TMPDIR") in ("/dev/shm", "/tmp"):
        notices.append(f"Using local TMPDIR at {os.environ['TMPDIR']} to avoid NFS temp-file cleanup issues.")

    ctx = collect_run_context(args, predictor, kind, df_train_full, df_val, df_test, caught_warnings, notices)

    leaderboard_html = "" if is_multimodal else build_leaderboard_html(predictor)
    inputs_html = ""
    ignored_features_html = "" if is_multimodal else build_ignored_features_html(predictor, df_train_full)

    presets_hparams_html = build_presets_hparams_html(predictor)
    warnings_html = build_warnings_html(caught_warnings, notices)
    repro_html = build_reproducibility_html(args, ctx, getattr(predictor, "path", None))

    transparency_blocks = "\n".join(
        [
            leaderboard_html,
            inputs_html,
            ignored_features_html,
            presets_hparams_html,
            warnings_html,
            repro_html,
        ]
    )

    full_html = assemble_full_html_report(
        summary_html,
        train_html,
        test_html_filled,
        plots,
        feature_text + folds_html + transparency_blocks,
    )
    with open(args.output_html, "w") as f:
        f.write(full_html)
    logger.info(f"Wrote HTML report → {args.output_html}")

    verify_outputs(
        [
            (args.output_csv, "CSV metrics"),
            (args.output_json, "JSON results"),
            (args.output_html, "HTML report"),
        ]
    )
    logger.info(f"Final working directory contents: {os.listdir('.')}")


if __name__ == "__main__":
    main()
