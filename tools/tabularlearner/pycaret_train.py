import argparse
import logging
import os

from pycaret_classification import ClassificationModelTrainer
from pycaret_regression import RegressionModelTrainer

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the input file")
    parser.add_argument("--target_col", help="Column number of the target")
    parser.add_argument("--output_dir", help="Path to the output directory")
    parser.add_argument(
        "--model_type",
        choices=["classification", "regression"],
        help="Type of the model",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=None,
        help="Train size for PyCaret setup",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=None,
        help="Normalize data for PyCaret setup",
    )
    parser.add_argument(
        "--feature_selection",
        action="store_true",
        default=None,
        help="Perform feature selection for PyCaret setup",
    )
    parser.add_argument(
        "--cross_validation",
        action="store_true",
        default=True,
        help="Enable cross-validation for PyCaret setup (default: enabled)",
    )
    parser.add_argument(
        "--no_cross_validation",
        action="store_true",
        default=None,
        help="Disable cross-validation for PyCaret setup",
    )
    parser.add_argument(
        "--cross_validation_folds",
        type=int,
        default=None,
        help="Number of cross-validation folds for PyCaret setup",
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        default=None,
        help="Remove outliers for PyCaret setup",
    )
    parser.add_argument(
        "--remove_multicollinearity",
        action="store_true",
        default=None,
        help="Remove multicollinearity for PyCaret setup",
    )
    parser.add_argument(
        "--polynomial_features",
        action="store_true",
        default=None,
        help="Generate polynomial features for PyCaret setup",
    )
    parser.add_argument(
        "--feature_interaction",
        action="store_true",
        default=None,
        help="Generate feature interactions for PyCaret setup",
    )
    parser.add_argument(
        "--feature_ratio",
        action="store_true",
        default=None,
        help="Generate feature ratios for PyCaret setup",
    )
    parser.add_argument(
        "--fix_imbalance",
        action="store_true",
        default=None,
        help="Fix class imbalance for PyCaret setup",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Selected models for training",
    )
    parser.add_argument(
        "--tune_model",
        action="store_true",
        default=False,
        help="Tune the best model hyperparameters after training",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to the test data file",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for PyCaret setup",
    )
    parser.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs; defaults to GALAXY_SLOTS or 1 if unset/invalid.",
    )
    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=None,
        help="Probability threshold for classification decision,",
    )
    parser.add_argument(
        "--threshold_mode",
        choices=["auto", "manual"],
        default="auto",
        help="Whether to optimize the binary classification threshold or use a manual value.",
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="F1",
        help="Metric used for automatic binary threshold optimization.",
    )
    parser.add_argument(
        "--best_model_metric",
        type=str,
        default=None,
        help="Metric used to select the best model (e.g. AUC, Accuracy, R2, RMSE).",
    )
    parser.add_argument(
        "--sample-id-column",
        type=str,
        default=None,
        help=(
            "Optional column name used to group samples during splitting "
            "to prevent data leakage (e.g., patient_id or slide_id)."
        ),
    )

    args = parser.parse_args()

    # Derive n_jobs from CLI or GALAXY_SLOTS env var
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        slots_str = os.environ.get("GALAXY_SLOTS")
        try:
            n_jobs = int(slots_str) if slots_str is not None else 1
        except ValueError:
            n_jobs = 1

    # Normalize cross-validation flags: enabled by default, with
    # --no_cross_validation as the explicit override.
    if args.no_cross_validation:
        args.cross_validation = False

    if args.cross_validation and args.cross_validation_folds is None:
        args.cross_validation_folds = 10

    # Build the model_kwargs dict from CLI args
    model_kwargs = {
        "train_size": args.train_size,
        "normalize": args.normalize,
        "feature_selection": args.feature_selection,
        "cross_validation": args.cross_validation,
        "cross_validation_folds": args.cross_validation_folds,
        "remove_outliers": args.remove_outliers,
        "remove_multicollinearity": args.remove_multicollinearity,
        "polynomial_features": args.polynomial_features,
        "feature_interaction": args.feature_interaction,
        "feature_ratio": args.feature_ratio,
        "fix_imbalance": args.fix_imbalance,
        "tune_model": args.tune_model,
        "n_jobs": n_jobs,
        "probability_threshold": args.probability_threshold,
        "threshold_mode": args.threshold_mode,
        "threshold_metric": args.threshold_metric,
        "best_model_metric": args.best_model_metric,
        "sample_id_column": args.sample_id_column,
    }
    LOG.info(f"Model kwargs: {model_kwargs}")

    # If the XML passed a comma-separated string in a single list element, split it out
    if args.models:
        model_kwargs["models"] = args.models[0].split(",")

    # Drop None entries so PyCaret uses its default values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    LOG.info(f"Model kwargs 2: {model_kwargs}")

    # Instantiate the appropriate trainer
    if args.model_type == "classification":
        trainer = ClassificationModelTrainer(
            args.input_file,
            args.target_col,
            args.output_dir,
            args.model_type,
            args.random_seed,
            args.test_file,
            **model_kwargs,
        )
    elif args.model_type == "regression":
        # regression doesn't support fix_imbalance
        model_kwargs.pop("fix_imbalance", None)
        model_kwargs.pop("probability_threshold", None)
        model_kwargs.pop("threshold_mode", None)
        model_kwargs.pop("threshold_metric", None)
        trainer = RegressionModelTrainer(
            args.input_file,
            args.target_col,
            args.output_dir,
            args.model_type,
            args.random_seed,
            args.test_file,
            **model_kwargs,
        )
    else:
        LOG.error("Invalid model type. Please choose 'classification' or 'regression'.")
        return

    trainer.run()


if __name__ == "__main__":
    main()
