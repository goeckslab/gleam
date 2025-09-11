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
    path_expander,
)
from plot_logic import (
    infer_problem_type,
    build_summary_html,
    build_test_html_and_plots,
    build_feature_html,
    assemble_full_html_report,
)
from metrics_logic import evaluate_all_transparency

# Transparency helpers (report_utils.py)
from report_utils import (
    collect_run_context,
    build_class_balance_html,
    build_leaderboard_html,
    build_ignored_features_html,
    build_presets_hparams_html,
    build_warnings_html,
    build_reproducibility_html,
    build_modalities_html,  # for multimodal runs
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
    """Use Tensor Cores-friendly matmul where supported (PyTorch 2+)."""
    try:
        torch.set_float32_matmul_precision("high")
        logger.info("Enabled torch float32 matmul precision = 'high' (Tensor Cores).")
    except Exception:
        pass


def ensure_local_tmp():
    """Prefer a local, non-NFS tmp to avoid '.nfs* device busy' cleanup warnings."""
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
            logger.info(f'✓ Output {desc}: {p} ({size:,} bytes)')
            os.chmod(p, 0o644)
        else:
            logger.error(f'✗ Output {desc} MISSING: {p}')
            ok = False
    if not ok:
        logger.error('Some outputs are missing!')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train & report an AutoGluon model')
    parser.add_argument('--input_csv_train', dest='train_csv', required=True)
    parser.add_argument('--input_csv_test', dest='test_csv', default=None)
    parser.add_argument('--target_column', dest='label_column', required=True)
    parser.add_argument('--output_csv', dest='output_csv', required=True)
    parser.add_argument('--output_json', dest='output_json', default='results.json')
    parser.add_argument('--output_html', dest='output_html', default='report.html')

    parser.add_argument('--image_column', dest='image_column', default=None)
    parser.add_argument('--images_zip', dest='images_zip', default=None)
    parser.add_argument('--image_folder', dest='image_folder', default=None)

    parser.add_argument('--time_limit', dest='time_limit', type=int, default=None)
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42)

    # Split knobs (configurable)
    parser.add_argument('--validation_size', type=float, default=0.125,
                        help='When fixed split contains {0,2}, fraction of 0 reassigned to 1.')
    parser.add_argument('--split_probabilities', type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        metavar=('train', 'val', 'test'),
                        help='Random split proportions when no split column exists.')
    parser.add_argument('--val_size_with_test', type=float, default=0.2,
                        help='Validation fraction when a separate test CSV is provided.')

    args = parser.parse_args()

    # Validate split args
    if not (0.0 <= args.validation_size <= 1.0):
        parser.error('--validation_size must be in [0, 1]')
    if len(args.split_probabilities) != 3 or abs(sum(args.split_probabilities) - 1.0) > 1e-6:
        parser.error('--split_probabilities must be three numbers summing to 1.0')
    if not (0.0 < args.val_size_with_test < 1.0):
        parser.error('--val_size_with_test must be in (0, 1)')

    # Debug info
    logger.info('=== Galaxy Tool Debug Info ===')
    logger.info(f'Working directory: {os.getcwd()}')
    logger.info(f'Command line arguments: {sys.argv}')
    logger.info(f'Parsed arguments: {vars(args)}')
    logger.info(f'Input train CSV exists: {os.path.exists(args.train_csv)}')
    if args.images_zip:
        logger.info(f'Images ZIP exists: {os.path.exists(args.images_zip)}')
    logger.info('=== End Debug Info ===')

    # Reproducibility + perf
    set_seeds(args.random_seed)
    ensure_local_tmp()
    enable_tensor_cores_if_available()

    # Images ZIP extraction (if any)
    base_folder = args.image_folder or os.getcwd()
    if args.images_zip and os.path.isfile(args.images_zip):
        if not args.image_column:
            logger.warning('Images ZIP provided but no image_column specified; ignoring ZIP.')
        else:
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(args.images_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            base_folder = extract_dir
            logger.info(f'Extracted images ZIP to {extract_dir}')

    # Load + split (centralized)
    try:
        df_train, df_val, df_test, df_train_full, label_col, image_col = load_and_split(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            label_column=args.label_column,
            image_column=args.image_column,
            random_seed=args.random_seed,
            validation_size=args.validation_size,
            split_probabilities=args.split_probabilities,
            val_size_with_test=args.val_size_with_test,
        )
        args.label_column = label_col
        args.image_column = image_col
    except Exception as e:
        logger.error(f'Failed to read/split input CSVs: {e}')
        sys.exit(1)

    # Verify required columns
    for df_, name in [(df_train_full, 'train'), (df_test, 'test')]:
        if args.label_column not in df_.columns:
            logger.error(f"Missing target column '{args.label_column}' in {name} CSV")
            sys.exit(1)
        if args.image_column and args.image_column not in df_.columns:
            logger.error(f"Missing image column '{args.image_column}' in {name} CSV")
            sys.exit(1)

    logger.info(f'Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test')

    # Expand image paths if using images
    if args.image_column:
        for df_ in (df_train, df_val, df_test):
            df_[args.image_column] = df_[args.image_column].apply(lambda p: path_expander(str(p), base_folder))

    # Capture warnings across run
    caught_warnings: List[str] = []

    def _warn_recorder(message, category, filename, lineno, file=None, line=None):
        try:
            caught_warnings.append(f"{category.__name__}: {message}")
        except Exception:
            pass

    warnings.showwarning = _warn_recorder
    warnings.filterwarnings("default")

    # Train
    if args.image_column:
        logger.info('Starting AutoGluon MultiModal training...')
        column_types = {args.image_column: 'image_path'}
        predictor = MultiModalPredictor(label=args.label_column, path=None)
        mm_hparams = {"optimization": {"seed": int(args.random_seed)}}
        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
            column_types=column_types,
            hyperparameters=mm_hparams,
            seed=int(args.random_seed),
        )
    else:
        logger.info('Starting AutoGluon Tabular training...')
        predictor = TabularPredictor(label=args.label_column, path=None)
        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
            seed=int(args.random_seed),
        )

    # Persist predictor path (for Galaxy packaging)
    try:
        pred_path = getattr(predictor, "path", None)
        if pred_path:
            with open("predictor_path.txt", "w") as pf:
                pf.write(str(pred_path))
            logger.info(f"Wrote predictor path → predictor_path.txt ({pred_path})")
    except Exception:
        logger.warning("Could not write predictor_path.txt")

    # Determine problem type
    kind = infer_problem_type(predictor, df_train_full, args.label_column)  # 'binary' | 'multiclass' | 'regression'
    logger.info(f'Inferred problem type: {kind}')

    # Evaluate with transparent metrics
    _, raw_metrics = evaluate_all_transparency(
        predictor=predictor,
        train_df=df_train,
        val_df=df_val,
        test_df=df_test,
        target_col=args.label_column,
        problem_type=kind,
    )

    # Write metrics CSV (flattened rows per phase, dynamic columns by task)
    all_keys = []
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

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.output_csv, index=False)
    logger.info(f'Wrote metrics CSV → {args.output_csv}')

    # JSON (fit_summary may not exist for MultiModal; handle gracefully)
    fit_summary_obj: Optional[dict] = None
    try:
        fit_summary_obj = predictor.fit_summary()
    except Exception:
        fit_summary_obj = None

    with open(args.output_json, 'w') as f:
        json.dump(
            {
                'train': raw_metrics.get('Train', {}),
                'val': raw_metrics.get('Validation', {}),
                'test': raw_metrics.get('Test', {}),
                'fit_summary': fit_summary_obj,
                'problem_type': kind,
                'predictor_path': getattr(predictor, "path", None),
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f'Wrote full JSON → {args.output_json}')

    # ---------------- HTML report ----------------
    tmpdir = tempfile.mkdtemp()

    # Summary
    summary_html = build_summary_html(
        predictor, args, kind,
        raw_metrics.get('Train', {}),
        raw_metrics.get('Validation', {}),
        raw_metrics.get('Test', {}),
        tmpdir
    )

    # Test section (reuse existing plot builder)
    test_html_template, plots = build_test_html_and_plots(predictor, kind, df_test, args.label_column, tmpdir)

    def _fmt_val(v):
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.6f}"
        return str(v)

    test_scores = raw_metrics.get('Test', {})
    metric_rows = ''.join(
        f'<tr><td>{k.replace("_"," ").replace("(TNR)","(TNR)").replace("(Sensitivity/TPR)", "(Sensitivity/TPR)")}</td>'
        f'<td>{_fmt_val(v)}</td></tr>'
        for k, v in test_scores.items()
    )
    test_html_filled = test_html_template.format(metric_rows)

    # MultiModal vs Tabular differences
    is_multimodal = isinstance(predictor, MultiModalPredictor)

    # Feature importance
    if is_multimodal:
        feature_html = (
            "<h3>Feature Importance</h3>"
            "<p>Permutation importance is not supported for MultiModalPredictor in this tool. "
            "For tabular-only runs, this section shows permutation importance.</p>"
        )
    else:
        feature_html = build_feature_html(predictor, df_test, args.label_column, tmpdir, args.random_seed)

    # Notices & warnings
    notices: List[str] = []
    notices.append("No presets specified; AutoGluon defaulted to 'medium' (fast prototyping).")
    if kind in ("binary", "multiclass") and len(df_val) < 10_000:
        notices.append("Decision threshold calibration disabled due to <10,000 validation rows (to avoid overfitting).")
    if any("bokeh is not installed" in w.lower() for w in caught_warnings):
        notices.append('To enable AutoGluon summary plots, install: pip install "bokeh==2.0.1"')
    if any("do not sum to one" in w.lower() for w in caught_warnings):
        notices.append("Detected probability-related warnings. Ensure probabilities (not labels) are used for PR-AUC/LogLoss.")
    if os.environ.get("TMPDIR") in ("/dev/shm", "/tmp"):
        notices.append(f"Using local TMPDIR at {os.environ['TMPDIR']} to avoid NFS temp-file cleanup issues.")

    # Extra transparency blocks
    ctx = collect_run_context(args, predictor, kind, df_train_full, df_val, df_test, caught_warnings, notices)
    class_balance_html = build_class_balance_html(df_train_full, args.label_column)

    # Leaderboard only for TabularPredictor
    leaderboard_html = "" if is_multimodal else build_leaderboard_html(predictor)

    # Inputs section: MultiModal shows modalities; Tabular shows ignored/unused (if detectable)
    if is_multimodal:
        inputs_html = build_modalities_html(predictor, df_train_full, args.label_column, args.image_column)
        ignored_features_html = ""
    else:
        inputs_html = ""
        ignored_features_html = build_ignored_features_html(predictor, df_train_full)

    presets_hparams_html = build_presets_hparams_html(predictor)
    warnings_html = build_warnings_html(caught_warnings, notices)
    repro_html = build_reproducibility_html(args, ctx, getattr(predictor, "path", None))

    transparency_blocks = "\n".join([
        class_balance_html,
        leaderboard_html,
        inputs_html,
        ignored_features_html,
        presets_hparams_html,
        warnings_html,
        repro_html,
    ])

    full_html = assemble_full_html_report(
        summary_html,
        test_html_filled,
        plots,
        feature_html + transparency_blocks
    )
    with open(args.output_html, 'w') as f:
        f.write(full_html)
    logger.info(f'Wrote HTML report → {args.output_html}')

    # Verify all outputs
    verify_outputs([
        (args.output_csv, 'CSV metrics'),
        (args.output_json, 'JSON results'),
        (args.output_html, 'HTML report'),
    ])

    logger.info(f'Final working directory contents: {os.listdir(".")}')


if __name__ == '__main__':
    main()
