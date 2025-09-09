import argparse
import json
import logging
import os
import random
import sys
import tempfile
import zipfile

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
    evaluate_all,
    infer_problem_type,
    build_summary_html,
    build_test_html_and_plots,
    build_feature_html,
    assemble_full_html_report,
)

# ------------- Logging -------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    # Reproducibility
    set_seeds(args.random_seed)

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

    # Train
    if args.image_column:
        logger.info('Starting AutoGluon MultiModal training...')
        column_types = {args.image_column: 'image_path'}
        predictor = MultiModalPredictor(label=args.label_column, path=None)
        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
            column_types=column_types,
        )
    else:
        logger.info('Starting AutoGluon Tabular training...')
        predictor = TabularPredictor(label=args.label_column, path=None)
        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
        )

    # Evaluate + plots
    kind = infer_problem_type(predictor, df_train_full, args.label_column)
    train_scores, val_scores, test_scores = evaluate_all(
        predictor, df_train, df_val, df_test, args.label_column, kind
    )

    # Write metrics CSV
    df_out = pd.DataFrame(
        [{'phase': 'train', **train_scores},
         {'phase': 'validation', **val_scores},
         {'phase': 'test', **test_scores}]
    )
    df_out.to_csv(args.output_csv, index=False)
    logger.info(f'Wrote metrics CSV → {args.output_csv}')

    # JSON
    with open(args.output_json, 'w') as f:
        json.dump(
            {
                'train': train_scores,
                'val': val_scores,
                'test': test_scores,
                'fit_summary': predictor.fit_summary(),
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f'Wrote full JSON → {args.output_json}')

    # HTML report
    tmpdir = tempfile.mkdtemp()
    summary_html = build_summary_html(predictor, args, kind, train_scores, val_scores, test_scores, tmpdir)

    # test plots + metrics table rows
    test_html_template, plots = build_test_html_and_plots(predictor, kind, df_test, args.label_column, tmpdir)
    metric_rows = ''.join(
        f'<tr><td>{k.replace("_"," ").title()}</td><td>{v:.4f}</td></tr>'
        for k, v in test_scores.items() if isinstance(v, (int, float))
    )
    test_html_filled = test_html_template.format(metric_rows)

    # feature importance
    feature_html = build_feature_html(predictor, df_test, args.label_column, tmpdir, args.random_seed)

    full_html = assemble_full_html_report(summary_html, test_html_filled, plots, feature_html)
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

