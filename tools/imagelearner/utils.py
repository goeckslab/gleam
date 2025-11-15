import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger("ImageLearner")


def load_metadata_table(file_path: Path) -> pd.DataFrame:
    """Load image metadata allowing either CSV or TSV delimiters."""
    logger.info("Loading metadata table from %s", file_path)
    return pd.read_csv(file_path, sep=None, engine="python")

def detect_output_type(test_stats):
    """Detects if the output type is 'binary' or 'category' based on test statistics."""
    label_stats = test_stats.get("label", {})
    if "mean_squared_error" in label_stats:
        return "regression"
    per_class = label_stats.get("per_class_stats", {})
    if len(per_class) == 2:
        return "binary"
    return "category"

def aug_parse(aug_string: str):
    """
    Parse comma-separated augmentation keys into Ludwig augmentation dicts.
    Raises ValueError on unknown key.
    """
    mapping = {
        "random_horizontal_flip": {"type": "random_horizontal_flip"},
        "random_vertical_flip": {"type": "random_vertical_flip"},
        "random_rotate": {"type": "random_rotate", "degree": 10},
        "random_blur": {"type": "random_blur", "kernel_size": 3},
        "random_brightness": {"type": "random_brightness", "min": 0.5, "max": 2.0},
        "random_contrast": {"type": "random_contrast", "min": 0.5, "max": 2.0},
    }
    aug_list = []
    for tok in aug_string.split(","):
        key = tok.strip()
        if not key:
            continue
        if key not in mapping:
            valid = ", ".join(mapping.keys())
            raise ValueError(f"Unknown augmentation '{key}'. Valid choices: {valid}")
        aug_list.append(mapping[key])
    return aug_list

def argument_checker(args, parser):
  if not 0.0 <= args.validation_size <= 1.0:
    parser.error("validation-size must be between 0.0 and 1.0")
  if not args.csv_file.is_file():
    parser.error(f"Metada file not found: {args.csv_file}")
  if not (args.image_zip.is_file() or args.image_zip.is_dir()):
    parser.error(f"ZIP or directory not found: {args.image_zip}")
  if args.augmentation is not None:
    try:
      augmentation_setup = aug_parse(args.augmentation)
      setattr(args, "augmentation", augmentation_setup)
    except ValueError as e:
      parser.error(str(e))

def parse_learning_rate(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def extract_metrics_from_json(
    train_stats: dict,
    test_stats: dict,
    output_type: str,
) -> dict:
    """Extracts relevant metrics from training and test statistics based on the output type."""
    metrics = {"training": {}, "validation": {}, "test": {}}

    def get_last_value(stats, key):
        val = stats.get(key)
        if isinstance(val, list) and val:
            return val[-1]
        elif isinstance(val, (int, float)):
            return val
        return None

    for split in ["training", "validation"]:
        split_stats = train_stats.get(split, {})
        if not split_stats:
            logger.warning("No statistics found for %s split", split)
            continue
        label_stats = split_stats.get("label", {})
        if not label_stats:
            logger.warning("No label statistics found for %s split", split)
            continue
        if output_type == "binary":
            metrics[split] = {
                "accuracy": get_last_value(label_stats, "accuracy"),
                "loss": get_last_value(label_stats, "loss"),
                "precision": get_last_value(label_stats, "precision"),
                "recall": get_last_value(label_stats, "recall"),
                "specificity": get_last_value(label_stats, "specificity"),
                "roc_auc": get_last_value(label_stats, "roc_auc"),
            }
        elif output_type == "regression":
            metrics[split] = {
                "loss": get_last_value(label_stats, "loss"),
                "mean_absolute_error": get_last_value(
                    label_stats, "mean_absolute_error"
                ),
                "mean_absolute_percentage_error": get_last_value(
                    label_stats, "mean_absolute_percentage_error"
                ),
                "mean_squared_error": get_last_value(label_stats, "mean_squared_error"),
                "root_mean_squared_error": get_last_value(
                    label_stats, "root_mean_squared_error"
                ),
                "root_mean_squared_percentage_error": get_last_value(
                    label_stats, "root_mean_squared_percentage_error"
                ),
                "r2": get_last_value(label_stats, "r2"),
            }
        else:
            metrics[split] = {
                "accuracy": get_last_value(label_stats, "accuracy"),
                "accuracy_micro": get_last_value(label_stats, "accuracy_micro"),
                "loss": get_last_value(label_stats, "loss"),
                "roc_auc": get_last_value(label_stats, "roc_auc"),
                "hits_at_k": get_last_value(label_stats, "hits_at_k"),
            }

    # Test metrics: dynamic extraction according to exclusions
    test_label_stats = test_stats.get("label", {})
    if not test_label_stats:
        logger.warning("No label statistics found for test split")
    else:
        combined_stats = test_stats.get("combined", {})
        overall_stats = test_label_stats.get("overall_stats", {})

        # Define exclusions
        if output_type == "binary":
            exclude = {"per_class_stats", "precision_recall_curve", "roc_curve"}
        else:
            exclude = {"per_class_stats", "confusion_matrix"}

        # 1. Get all scalar test_label_stats not excluded
        test_metrics = {}
        for k, v in test_label_stats.items():
            if k in exclude:
                continue
            if k == "overall_stats":
                continue
            if isinstance(v, (int, float, str, bool)):
                test_metrics[k] = v

        # 2. Add overall_stats (flattened)
        for k, v in overall_stats.items():
            test_metrics[k] = v

        # 3. Optionally include combined/loss if present and not already
        if "loss" in combined_stats and "loss" not in test_metrics:
            test_metrics["loss"] = combined_stats["loss"]
        metrics["test"] = test_metrics
    return metrics
