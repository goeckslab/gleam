import argparse
import logging

from pycaret_classification import ClassificationModelTrainer
from pycaret_regression import RegressionModelTrainer

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the input file")
    parser.add_argument("--target_col", help="Column number of the target")
    parser.add_argument("--output_dir",
                        help="Path to the output directory")
    parser.add_argument("--model_type",
                        choices=["classification", "regression"],
                        help="Type of the model")
    parser.add_argument("--train_size", type=float,
                        default=None,
                        help="Train size for PyCaret setup")
    parser.add_argument("--normalize", action="store_true",
                        default=None,
                        help="Normalize data for PyCaret setup")
    parser.add_argument("--feature_selection", action="store_true",
                        default=None,
                        help="Perform feature selection for PyCaret setup")
    parser.add_argument("--cross_validation", action="store_true",
                        default=None,
                        help="Perform cross-validation for PyCaret setup")
    parser.add_argument("--no_cross_validation", action="store_true",
                        default=None,
                        help="Don't perform cross-validation for PyCaret setup")
    parser.add_argument("--cross_validation_folds", type=int,
                        default=None,
                        help="Number of cross-validation folds \
                          for PyCaret setup")
    parser.add_argument("--remove_outliers", action="store_true",
                        default=None,
                        help="Remove outliers for PyCaret setup")
    parser.add_argument("--remove_multicollinearity", action="store_true",
                        default=None,
                        help="Remove multicollinearity for PyCaret setup")
    parser.add_argument("--polynomial_features", action="store_true",
                        default=None,
                        help="Generate polynomial features for PyCaret setup")
    parser.add_argument("--feature_interaction", action="store_true",
                        default=None,
                        help="Generate feature interactions for PyCaret setup")
    parser.add_argument("--feature_ratio", action="store_true",
                        default=None,
                        help="Generate feature ratios for PyCaret setup")
    parser.add_argument("--fix_imbalance", action="store_true",
                        default=None,
                        help="Fix class imbalance for PyCaret setup")
    parser.add_argument("--models", nargs='+',
                        default=None,
                        help="Selected models for training")
    parser.add_argument("--random_seed", type=int,
                        default=42,
                        help="Random seed for PyCaret setup")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to the test data file")

    args = parser.parse_args()

    cross_validation = True
    if args.no_cross_validation:
        cross_validation = False

    model_kwargs = {
        "train_size": args.train_size,
        "normalize": args.normalize,
        "feature_selection": args.feature_selection,
        "cross_validation": cross_validation,
        "cross_validation_folds": args.cross_validation_folds,
        "remove_outliers": args.remove_outliers,
        "remove_multicollinearity": args.remove_multicollinearity,
        "polynomial_features": args.polynomial_features,
        "feature_interaction": args.feature_interaction,
        "feature_ratio": args.feature_ratio,
        "fix_imbalance": args.fix_imbalance,
    }
    LOG.info(f"Model kwargs: {model_kwargs}")

    # Remove None values from model_kwargs

    LOG.info(f"Model kwargs 2: {model_kwargs}")
    if args.models:
        model_kwargs["models"] = args.models[0].split(",")

    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    if args.model_type == "classification":
        trainer = ClassificationModelTrainer(
            args.input_file,
            args.target_col,
            args.output_dir,
            args.model_type,
            args.random_seed,
            args.test_file,
            **model_kwargs)
    elif args.model_type == "regression":
        if "fix_imbalance" in model_kwargs:
            del model_kwargs["fix_imbalance"]
        trainer = RegressionModelTrainer(
            args.input_file,
            args.target_col,
            args.output_dir,
            args.model_type,
            args.random_seed,
            args.test_file,
            **model_kwargs)
    else:
        LOG.error("Invalid model type. Please choose \
                  'classification' or 'regression'.")
        return
    trainer.run()


if __name__ == "__main__":
    main()
