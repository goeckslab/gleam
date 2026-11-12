import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from constants import LUDWIG_BOOL_FALSE_STRS, LUDWIG_BOOL_TRUE_STRS, MAX_ENSEMBLE_MEMBERS

logger = logging.getLogger("ImageLearner")


def load_metadata_table(file_path: Path) -> pd.DataFrame:
    """Load image metadata allowing either CSV or TSV delimiters."""
    logger.info("Loading metadata table from %s", file_path)
    return pd.read_csv(file_path, sep=None, engine="python")


def nonzero_binary_label(labels: Iterable[object]) -> Optional[str]:
    """Ludwig's positive class among two numeric labels, as the given string.

    Ludwig casts a numeric binary target to bool (zero -> False, nonzero ->
    True), so when the labels are two finite numbers and exactly one is zero,
    the nonzero one is the positive class. Returns None for anything else.
    """
    numbers = {}
    for label in labels:
        try:
            numbers[str(label)] = float(label)
        except (TypeError, ValueError):
            return None
    zeros = [name for name, number in numbers.items() if number == 0]
    if len(numbers) != 2 or len(zeros) != 1 or not all(map(math.isfinite, numbers.values())):
        return None
    return next(name for name in numbers if name != zeros[0])


def ludwig_text_binary_truth(labels: Iterable[object]) -> Dict[str, bool]:
    """How Ludwig 0.10.1 maps two text labels to booleans (binary_feature.get_feature_meta).

    Labels in Ludwig's boolean sets (case-insensitive) take their value from
    them. If either label is in neither set, the alphabetically first label is
    the fallback True label: the other label is True only if the sets say so.
    Both labels can end up with the same value, for example "yes"/"true" or
    "maybe"/"yes", and the model then cannot tell the classes apart.
    """
    names = [str(label) for label in labels]

    def from_sets(name: str) -> Optional[bool]:
        lowered = name.lower()
        if lowered in LUDWIG_BOOL_TRUE_STRS:
            return True
        if lowered in LUDWIG_BOOL_FALSE_STRS:
            return False
        return None

    strict = {name: from_sets(name) for name in names}
    if all(value is not None for value in strict.values()):
        return strict
    fallback_true = sorted(names)[0]
    return {name: (strict[name] if strict[name] is not None else name == fallback_true) for name in names}


_POSITIVE_CLASS_HINTS = ("event", "true", "positive", "pos", "1")


def positive_class_name(class_names: Iterable[object]) -> Optional[str]:
    """The positive class of a binary task; the report and the ensemble must agree on it.

    Two numeric classes with exactly one zero follow Ludwig (the nonzero label
    is positive); otherwise the first name in sorted order holding a hint such
    as "true" or "1" decides, then the last name in sorted order.
    """
    names = [str(name) for name in class_names]
    if not names:
        return None
    nonzero = nonzero_binary_label(names)
    if nonzero is not None:
        return nonzero
    ordered = sorted(names)
    for name in ordered:
        if any(hint in name.lower() for hint in _POSITIVE_CLASS_HINTS):
            return name
    return ordered[-1]


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
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be between 0.0 and 1.0")
    if getattr(args, "cross_validation", False) and getattr(args, "num_folds", 5) < 2:
        parser.error("num-folds must be at least 2 when cross-validation is enabled")
    if getattr(args, "num_seeds", 1) < 1:
        parser.error("num-seeds must be at least 1")
    folds = getattr(args, "num_folds", 5) if getattr(args, "cross_validation", False) else 1
    members = folds * getattr(args, "num_seeds", 1)
    if members > MAX_ENSEMBLE_MEMBERS:
        parser.error(
            f"folds x seeds = {members} models; at most {MAX_ENSEMBLE_MEMBERS} are allowed because members "
            "train one after another. Lower the number of folds or seeds."
        )
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

    metrics["test"] = extract_test_metrics(test_stats, output_type)
    return metrics


def extract_test_metrics(test_stats: dict, output_type: str) -> dict:
    """Flatten the scalar test metrics of Ludwig-shaped test statistics."""
    test_label_stats = test_stats.get("label", {})
    if not test_label_stats:
        logger.warning("No label statistics found for test split")
        return {}

    combined_stats = test_stats.get("combined", {})
    overall_stats = test_label_stats.get("overall_stats", {})

    # Define exclusions
    if output_type == "binary":
        exclude = {"per_class_stats", "precision_recall_curve", "roc_curve"}
    else:
        exclude = {"per_class_stats", "confusion_matrix"}
        # Ludwig 0.10.1 bug (ludwig/utils/eval_utils.py::ConfusionMatrix.stats):
        # "avg_precision_weighted" and "avg_recall_weighted" are computed with
        # average="micro", so they silently duplicate the micro values instead of
        # being support-weighted. "token_accuracy" (sklearn accuracy_score) is an
        # exact duplicate of accuracy_micro under a name that is meaningless for
        # image classification. Drop all three from the report.
        exclude |= {
            "avg_precision_weighted",
            "avg_recall_weighted",
            "token_accuracy",
        }

    # 1. Get all scalar test_label_stats not excluded
    test_metrics = {}
    for k, v in test_label_stats.items():
        if k in exclude:
            continue
        if k == "overall_stats":
            continue
        if isinstance(v, (int, float, str, bool)):
            test_metrics[k] = v

    # 2. Add overall_stats (flattened), applying the same exclusions
    for k, v in overall_stats.items():
        if k in exclude:
            continue
        test_metrics[k] = v

    # 3. Optionally include combined/loss if present and not already
    if "loss" in combined_stats and "loss" not in test_metrics:
        test_metrics["loss"] = combined_stats["loss"]
    return test_metrics
