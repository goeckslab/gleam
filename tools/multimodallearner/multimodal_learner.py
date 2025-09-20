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
from autogluon.tabular import TabularPredictor

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
    parser = argparse.ArgumentParser(description="Train & report an AutoGluon model")
    parser.add_argument("--input_csv_train", dest="train_csv", required=True)
    parser.add_argument("--input_csv_test", dest="test_csv", default=None)
    parser.add_argument("--target_column", dest="label_column", required=True)
    parser.add_argument("--output_csv", dest="output_csv", required=True)
    parser.add_argument("--output_json", dest="output_json", default="results.json")
    parser.add_argument("--output_html", dest="output_html", default="report.html")

    # Images (lists + legacy)
    parser.add_argument("--image_columns", dest="image_columns", nargs="+", default=None)
    parser.add_argument("--image_column", dest="image_column", default=None)
    parser.add_argument("--images_zips", dest="images_zips", nargs="*", default=None)
    parser.add_argument("--images_zip", dest="images_zip", default=None)
    parser.add_argument("--image_folders", dest="image_folders", nargs="*", default=None)

    # Threshold only for Test
    parser.add_argument("--threshold", dest="threshold", type=float, default=None)

    parser.add_argument("--time_limit", dest="time_limit", type=int, default=None)
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=42)

    # Split knobs
    parser.add_argument("--validation_size", type=float, default=0.125)
    parser.add_argument("--split_probabilities", type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        metavar=("train", "val", "test"))
    parser.add_argument("--val_size_with_test", type=float, default=0.2)

    # Cheat-sheet knobs
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument("--eval_metric", default=None)
    parser.add_argument("--excluded_model_types", nargs="*", default=None)
    parser.add_argument("--num_bag_folds", type=int, default=None)
    parser.add_argument("--num_stack_levels", type=int, default=None)
    parser.add_argument("--refit_full", action="store_true")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--hyperparameters", default=None)

    args = parser.parse_args()

    # Normalize legacy flags
    img_cols = args.image_columns or ([args.image_column] if args.image_column else None)
    args.image_columns = img_cols

    zips = []
    if args.images_zips: zips.extend(args.images_zips)
    if args.images_zip:  zips.append(args.images_zip)
    args.images_zips = [z for z in zips if z]

    args.image_folders = [d for d in (args.image_folders or []) if d and os.path.isdir(d)]

    # Validate split args
    if not (0.0 <= args.validation_size <= 1.0):
        parser.error("--validation_size must be in [0, 1]")
    if len(args.split_probabilities) != 3 or abs(sum(args.split_probabilities) - 1.0) > 1e-6:
        parser.error("--split_probabilities must be three numbers summing to 1.0")
    if not (0.0 < args.val_size_with_test < 1.0):
        parser.error("--val_size_with_test must be in (0, 1)")

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

    # Build base folders (extract zips first, highest priority)
    base_folders: List[str] = list(args.image_folders or [])
    for z in (args.images_zips or []):
        if os.path.isfile(z):
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            base_folders.insert(0, extract_dir)
            logger.info(f"Extracted images ZIP to {extract_dir}")
    base_folders.append(os.getcwd())

    # Load + split
    try:
        df_train, df_val, df_test, df_train_full, label_col, image_cols = load_and_split(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            label_column=args.label_column,
            image_columns=args.image_columns,
            random_seed=args.random_seed,
            validation_size=args.validation_size,
            split_probabilities=args.split_probabilities,
            val_size_with_test=args.val_size_with_test,
        )
        args.label_column = label_col
        args.image_columns = image_cols
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

    # Expand image paths
    if args.image_columns:
        for df_ in (df_train, df_val, df_test):
            for c in args.image_columns:
                df_[c] = df_[c].astype(str).apply(lambda p: path_expander_any(p, base_folders))

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

    # Build hparams & train
    mm_hparams = build_mm_hparams(args, df_train, args.image_columns)
    predictor = train_predictor(args, df_train, df_val, args.image_columns, mm_hparams)

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
    tabular_cols = [c for c in df_train_full.columns if c not in exclude_cols]
    tabular_count = len(tabular_cols)

    if isinstance(predictor, MultiModalPredictor):
        modalities_inputs_text = "MultiModalPredictor (images + tabular)"
    else:
        modalities_inputs_text = "TabularPredictor (tabular)"

    presets_used = " ".join(args.presets) if args.presets else "AutoGluon default"
    calib_text = "disabled (due to <10,000 validation rows (to avoid overfitting))" if len(df_val) < 10_000 else "enabled"
    threshold_val = "None" if args.threshold is None else f"{float(args.threshold):.3f}"
    time_limit_val = "None" if args.time_limit is None else str(int(args.time_limit))
    arch_str = get_model_architecture(predictor)

    extra_run_rows = [
        ("Model architecture", arch_str),
        ("Modalities & Inputs", modalities_inputs_text),
        ("Label column", label_col),
        ("Image columns", img_cols_display),
        ("Tabular columns", str(tabular_count)),
        ("Presets", presets_used),
        ("Eval metric", args.eval_metric or "AutoGluon default"),
        ("Decision threshold calibration", calib_text),
        ("Decision threshold (Test only)", threshold_val),
        ("Seed", str(int(args.random_seed))),
        ("time limit(s)", time_limit_val),
    ]
    if not isinstance(predictor, MultiModalPredictor):
        extra_run_rows.extend([
            ("Excluded model types", " ".join(args.excluded_model_types) if args.excluded_model_types else "—"),
            ("num_bag_folds", str(args.num_bag_folds) if args.num_bag_folds is not None else "—"),
            ("num_stack_levels", str(args.num_stack_levels) if args.num_stack_levels is not None else "—"),
            ("refit_full", "yes" if args.refit_full else "no"),
        ])

    class_balance_block_html = build_class_balance_html(df_train_full, label_col)

    summary_perf_table_html = build_model_performance_summary_table(
        train_scores=raw_metrics.get("Train", {}),
        val_scores=raw_metrics.get("Validation", {}),
        test_scores=raw_metrics.get("Test", {}),
        include_test=True,
        title=None,
        show_title=False,
    )
    summary_html = build_summary_html(
        predictor=predictor,
        df_train=df_train_full,
        df_val=df_val,
        df_test=df_test,
        label_column=args.label_column,
        extra_run_rows=extra_run_rows,
        class_balance_html=class_balance_block_html,
        perf_table_html=summary_perf_table_html,
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
        feature_html = (
            "<h3>Feature Importance</h3>"
            "<p>Permutation importance is not supported for MultiModalPredictor in this tool. "
            "For tabular-only runs, this section shows permutation importance.</p>"
        )
    else:
        feature_html = build_feature_html(predictor, df_test, args.label_column, tmpdir, args.random_seed)

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
        feature_html + transparency_blocks,
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