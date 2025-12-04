#!/usr/bin/env python
"""
Main entrypoint for AutoGluon multimodal training wrapper.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import pandas as pd
from metrics_logic import aggregate_metrics
from plot_logic import infer_problem_type
from report_utils import write_outputs
from sklearn.model_selection import KFold, StratifiedKFold
from split_logic import split_dataset
from test_pipeline import run_autogluon_test_experiment
from training_pipeline import autogluon_hyperparameters, handle_missing_images, run_autogluon_experiment
# ------------------------------------------------------------------
# Local imports (your split utilities)
# ------------------------------------------------------------------
from utils import (
    absolute_path_expander,
    enable_deterministic_mode,
    enable_tensor_cores_if_available,
    ensure_local_tmp,
    load_file,
    prepare_image_search_dirs,
    set_seeds,
    str2bool,
)

# ------------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Argument parsing (unchanged from your original, only minor fixes)
# ------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train & report an AutoGluon model")

    parser.add_argument("--input_csv_train", dest="train_dataset", required=True)
    parser.add_argument("--input_csv_test", dest="test_dataset", default=None)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--output_json", default="results.json")
    parser.add_argument("--output_html", default="report.html")
    parser.add_argument("--output_config", default=None)
    parser.add_argument("--images_zip", nargs="*", default=None,
                        help="One or more ZIP files that contain image assets")
    parser.add_argument("--missing_image_strategy", default="false",
                        help="true/false: remove rows with missing images or use placeholder")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enable deterministic algorithms to reduce run-to-run variance")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cross_validation", type=str, default="false")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--backbone_image", type=str, default="swin_base_patch4_window7_224")
    parser.add_argument("--backbone_text", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--split_probabilities", type=float, nargs=3,
                        default=[0.7, 0.1, 0.2], metavar=("train", "val", "test"))
    parser.add_argument("--preset", choices=["medium_quality", "high_quality", "best_quality"],
                        default="medium_quality")
    parser.add_argument("--eval_metric", default="roc_auc")
    parser.add_argument("--hyperparameters", default=None)

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown CLI tokens: %s", unknown)

    # -------------------------- Validation --------------------------
    if not (0.0 <= args.validation_size <= 1.0):
        parser.error("--validation_size must be in [0, 1]")
    if len(args.split_probabilities) != 3 or abs(sum(args.split_probabilities) - 1.0) > 1e-6:
        parser.error("--split_probabilities must be three numbers summing to 1.0")
    if args.cross_validation.lower() == "true" and (args.num_folds < 2):
        parser.error("--num_folds must be >= 2 when --cross_validation is true")

    return args


def run_cross_validation(
    args,
    df_full: pd.DataFrame,
    test_dataset: Optional[pd.DataFrame],
    image_cols: List[str],
    ag_config: dict,
):
    """Cross-validation loop returning aggregated metrics and last predictor."""
    df_full = df_full.drop(columns=["split"], errors="ignore")
    y = df_full[args.target_column]
    try:
        use_stratified = y.dtype == object or y.nunique() <= 20
    except Exception:
        use_stratified = False

    kf = StratifiedKFold(n_splits=int(args.num_folds), shuffle=True, random_state=int(args.random_seed)) if use_stratified else KFold(n_splits=int(args.num_folds), shuffle=True, random_state=int(args.random_seed))

    raw_folds = []
    ag_folds = []
    folds_info = []
    last_predictor = None
    last_data_ctx = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_full, y if use_stratified else None), start=1):
        logger.info(f"CV fold {fold_idx}/{args.num_folds}")
        df_tr = df_full.iloc[train_idx].copy()
        df_va = df_full.iloc[val_idx].copy()

        df_tr["split"] = "train"
        df_va["split"] = "val"
        fold_dataset = pd.concat([df_tr, df_va], ignore_index=True)

        predictor_fold, data_ctx = run_autogluon_experiment(
            train_dataset=fold_dataset,
            test_dataset=test_dataset,
            target_column=args.target_column,
            image_columns=image_cols,
            ag_config=ag_config,
        )
        last_predictor = predictor_fold
        last_data_ctx = data_ctx
        problem_type = infer_problem_type(predictor_fold, df_tr, args.target_column)
        eval_results = run_autogluon_test_experiment(
            predictor=predictor_fold,
            data_ctx=data_ctx,
            target_column=args.target_column,
            eval_metric=args.eval_metric,
            ag_config=ag_config,
            problem_type=problem_type,
        )

        raw_metrics_fold = eval_results.get("raw_metrics", {})
        ag_by_split_fold = eval_results.get("ag_eval", {})
        raw_folds.append(raw_metrics_fold)
        ag_folds.append(ag_by_split_fold)
        folds_info.append(
            {
                "fold": int(fold_idx),
                "predictor_path": getattr(predictor_fold, "path", None),
                "raw_metrics": raw_metrics_fold,
                "ag_eval": ag_by_split_fold,
            }
        )

    raw_metrics_mean, raw_metrics_std = aggregate_metrics(raw_folds)
    ag_by_split_mean, ag_by_split_std = aggregate_metrics(ag_folds)
    return (
        last_predictor,
        raw_metrics_mean,
        ag_by_split_mean,
        raw_folds,
        ag_folds,
        raw_metrics_std,
        ag_by_split_std,
        folds_info,
        last_data_ctx,
    )


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------
    logger.info("=== AutoGluon Training Wrapper Started ===")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    logger.info(f"Parsed args: {vars(args)}")

    # ------------------------------------------------------------------
    # Reproducibility & performance
    # ------------------------------------------------------------------
    set_seeds(args.random_seed)
    if args.deterministic:
        enable_deterministic_mode(args.random_seed)
        logger.info("Deterministic mode enabled (seed=%s)", args.random_seed)
    ensure_local_tmp()
    enable_tensor_cores_if_available()

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    train_dataset = load_file(args.train_dataset)
    test_dataset = load_file(args.test_dataset) if args.test_dataset else None

    logger.info(f"Train dataset loaded: {len(train_dataset)} rows")
    if test_dataset is not None:
        logger.info(f"Test dataset loaded: {len(test_dataset)} rows")

    # ------------------------------------------------------------------
    # Resolve target column by name; if Galaxy passed a numeric index,
    # translate it to the corresponding header so downstream checks pass.
    # Galaxy's data_column widget is 1-based.
    # ------------------------------------------------------------------
    if args.target_column not in train_dataset.columns and str(args.target_column).isdigit():
        idx = int(args.target_column) - 1
        if 0 <= idx < len(train_dataset.columns):
            resolved = train_dataset.columns[idx]
            logger.info(f"Target column '{args.target_column}' not found; using column #{idx + 1} header '{resolved}' instead.")
            args.target_column = resolved
        else:
            logger.error(f"Numeric target index '{args.target_column}' is out of range for dataset with {len(train_dataset.columns)} columns.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Image handling (ZIP extraction + absolute path expansion)
    # ------------------------------------------------------------------
    extracted_imgs_path = prepare_image_search_dirs(args)

    image_cols = absolute_path_expander(train_dataset, extracted_imgs_path, None)
    if test_dataset is not None:
        absolute_path_expander(test_dataset, extracted_imgs_path, image_cols)

    # ------------------------------------------------------------------
    # Handle missing images
    # ------------------------------------------------------------------
    train_dataset = handle_missing_images(
        train_dataset,
        image_columns=image_cols,
        strategy=args.missing_image_strategy,
    )
    if test_dataset is not None:
        test_dataset = handle_missing_images(
            test_dataset,
            image_columns=image_cols,
            strategy=args.missing_image_strategy,
        )

    logger.info(f"After cleanup → train: {len(train_dataset)}, test: {len(test_dataset) if test_dataset is not None else 0}")

    # ------------------------------------------------------------------
    # Dataset splitting logic (adds 'split' column to train_dataset)
    # ------------------------------------------------------------------
    split_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        target_column=args.target_column,
        split_probabilities=args.split_probabilities,
        validation_size=args.validation_size,
        random_seed=args.random_seed,
    )

    logger.info("Preprocessing complete — ready for AutoGluon training!")
    logger.info(f"Final split counts:\n{train_dataset['split'].value_counts().sort_index()}")

    # Verify target/image/text columns exist
    if args.target_column not in train_dataset.columns:
        logger.error(f"Target column '{args.target_column}' not found in training data.")
        sys.exit(1)
    if test_dataset is not None and args.target_column not in test_dataset.columns:
        logger.error(f"Target column '{args.target_column}' not found in test data.")
        sys.exit(1)

    # Threshold is only meaningful for binary classification; ignore otherwise.
    threshold_for_run = args.threshold
    unique_labels = None
    target_looks_binary = False
    try:
        unique_labels = train_dataset[args.target_column].nunique(dropna=True)
        target_looks_binary = unique_labels == 2
    except Exception:
        logger.warning("Could not inspect target column '%s' for threshold validation; proceeding without binary check.", args.target_column)

    if threshold_for_run is not None:
        if target_looks_binary:
            threshold_for_run = float(threshold_for_run)
            logger.info("Applying custom decision threshold %.4f for binary evaluation.", threshold_for_run)
        else:
            logger.warning(
                "Threshold %.3f provided but target '%s' does not appear binary (unique labels=%s); ignoring threshold.",
                threshold_for_run,
                args.target_column,
                unique_labels if unique_labels is not None else "unknown",
            )
            threshold_for_run = None
    args.threshold = threshold_for_run
    # Image columns are auto-inferred; image_cols already resolved to absolute paths.
    # ------------------------------------------------------------------
    # Build AutoGluon configuration from CLI knobs
    # ------------------------------------------------------------------
    ag_config = autogluon_hyperparameters(
        threshold=args.threshold,
        time_limit=args.time_limit,
        random_seed=args.random_seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        backbone_image=args.backbone_image,
        backbone_text=args.backbone_text,
        preset=args.preset,
        eval_metric=args.eval_metric,
        hyperparameters=args.hyperparameters,
    )
    logger.info(f"AutoGluon config prepared: fit={ag_config.get('fit')}, hyperparameters keys={list(ag_config.get('hyperparameters', {}).keys())}")

    cv_enabled = str2bool(args.cross_validation)
    if cv_enabled:
        (
            predictor,
            raw_metrics,
            ag_by_split,
            raw_folds,
            ag_folds,
            raw_metrics_std,
            ag_by_split_std,
            folds_info,
            data_ctx,
        ) = run_cross_validation(
            args=args,
            df_full=train_dataset,
            test_dataset=test_dataset,
            image_cols=image_cols,
            ag_config=ag_config,
        )
        if predictor is None:
            logger.error("All CV folds failed. Exiting.")
            sys.exit(1)
        eval_results = {
            "raw_metrics": raw_metrics,
            "ag_eval": ag_by_split,
            "fit_summary": None,
        }
    else:
        predictor, data_ctx = run_autogluon_experiment(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            target_column=args.target_column,
            image_columns=image_cols,
            ag_config=ag_config,
        )
        logger.info("AutoGluon training finished. Model path: %s", getattr(predictor, "path", None))

        # Evaluate predictor on Train/Val/Test splits
        problem_type = infer_problem_type(predictor, train_dataset, args.target_column)
        eval_results = run_autogluon_test_experiment(
            predictor=predictor,
            data_ctx=data_ctx,
            target_column=args.target_column,
            eval_metric=args.eval_metric,
            ag_config=ag_config,
            problem_type=problem_type,
        )
        raw_metrics = eval_results.get("raw_metrics", {})
        ag_by_split = eval_results.get("ag_eval", {})
        raw_folds = ag_folds = raw_metrics_std = ag_by_split_std = None

    logger.info("Transparent metrics by split: %s", eval_results["raw_metrics"])
    logger.info("AutoGluon evaluate() by split: %s", eval_results["ag_eval"])

    if "problem_type" in eval_results:
        problem_type_final = eval_results["problem_type"]
    else:
        problem_type_final = infer_problem_type(predictor, train_dataset, args.target_column)

    write_outputs(
        args=args,
        predictor=predictor,
        problem_type=problem_type_final,
        eval_results=eval_results,
        data_ctx=data_ctx,
        raw_folds=raw_folds,
        ag_folds=ag_folds,
        raw_metrics_std=raw_metrics_std,
        ag_by_split_std=ag_by_split_std,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    # Quiet noisy image parsing logs (e.g., PIL.PngImagePlugin debug streams)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    main()
