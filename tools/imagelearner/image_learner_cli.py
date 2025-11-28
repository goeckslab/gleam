import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
from constants import MODEL_ENCODER_TEMPLATES
from image_workflow import ImageLearnerCLI
from ludwig_backend import LudwigDirectBackend
from split_data import SplitProbAction
from utils import argument_checker, parse_learning_rate

# Set matplotlib backend after imports
matplotlib.use('Agg')

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ImageLearner")


def main():
    parser = argparse.ArgumentParser(
        description="Image Classification Learner with Pluggable Backends",
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        type=Path,
        help="Path to the input metadata file (CSV, TSV, etc)",
    )
    parser.add_argument(
        "--image-zip",
        required=True,
        type=Path,
        help="Path to the images ZIP or a directory containing images",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=MODEL_ENCODER_TEMPLATES.keys(),
        help="Which model template to use",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use pretrained weights for the model",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Enable fine-tuning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (None = auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("learner_output"),
        help="Where to write outputs",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Fraction for validation (0.0–1.0)",
    )
    parser.add_argument(
        "--preprocessing-num-processes",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="CPU processes for data prep",
    )
    parser.add_argument(
        "--split-probabilities",
        type=float,
        nargs=3,
        metavar=("train", "val", "test"),
        action=SplitProbAction,
        default=[0.7, 0.1, 0.2],
        help=(
            "Random split proportions (e.g., 0.7 0.1 0.2).Only used if no split column."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting (default: 42)",
    )
    parser.add_argument(
        "--learning-rate",
        type=parse_learning_rate,
        default=None,
        help="Learning rate. If not provided, Ludwig will auto-select it.",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        help=(
            "Comma-separated list (in order) of any of: "
            "random_horizontal_flip, random_vertical_flip, random_rotate, "
            "random_blur, random_brightness, random_contrast. "
            "E.g. --augmentation random_horizontal_flip,random_rotate"
        ),
    )
    parser.add_argument(
        "--image-resize",
        type=str,
        choices=[
            "original", "96x96", "128x128", "160x160", "192x192", "220x220",
            "224x224", "256x256", "299x299", "320x320", "384x384", "448x448", "512x512"
        ],
        default="original",
        help="Image resize option. 'original' keeps images as-is, other options resize to specified dimensions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Decision threshold for binary classification (0.0–1.0)."
            "Overrides default 0.5."
        ),
    )
    parser.add_argument(
        "--validation-metric",
        type=str,
        default="roc_auc",
        choices=[
            "accuracy",
            "loss",
            "roc_auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "log_loss",
            "pearson_r",
            "mae",
            "mse",
            "rmse",
            "mape",
            "r2",
            "explained_variance",
        ],
        help="Metric Ludwig uses to select the best model during training/validation.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Name of the target/label column in the metadata file (defaults to 'label').",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default=None,
        help="Name of the image column in the metadata file (defaults to 'image_path').",
    )

    args = parser.parse_args()

    argument_checker(args, parser)

    backend_instance = LudwigDirectBackend()
    orchestrator = ImageLearnerCLI(args, backend_instance)

    exit_code = 0
    try:
        orchestrator.run()
        logger.info("Main script finished successfully.")
    except Exception as e:
        logger.error(f"Main script failed.{e}")
        exit_code = 1
    finally:
        sys.exit(exit_code)


if __name__ == "__main__":
    try:
        import ludwig

        logger.debug(f"Found Ludwig version: {ludwig.globals.LUDWIG_VERSION}")
    except ImportError:
        logger.error(
            "Ludwig library not found. Please ensure Ludwig is installed "
            "('pip install ludwig[image]')"
        )
        sys.exit(1)

    main()
