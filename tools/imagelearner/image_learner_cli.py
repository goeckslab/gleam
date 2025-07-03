import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import pandas as pd
import pandas.api.types as ptypes
import yaml
from constants import (
    IMAGE_PATH_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    METRIC_DISPLAY_NAMES,
    MODEL_ENCODER_TEMPLATES,
    SPLIT_COLUMN_NAME,
    TEMP_CONFIG_FILENAME,
    TEMP_CSV_FILENAME,
    TEMP_DIR_PREFIX
)
from ludwig.globals import (
    DESCRIPTION_FILE_NAME,
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME,
)
from ludwig.utils.data_utils import get_split_path
from ludwig.visualize import get_visualizations_registry
from sklearn.model_selection import train_test_split
from utils import (
    build_tabbed_html,
    encode_image_to_base64,
    get_html_closing,
    get_html_template,
    get_metrics_help_modal
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger("ImageLearner")


def format_config_table_html(
    config: dict,
    split_info: Optional[str] = None,
    training_progress: dict = None,
) -> str:
    display_keys = [
        "task_type",
        "model_name",
        "epochs",
        "batch_size",
        "fine_tune",
        "use_pretrained",
        "learning_rate",
        "random_seed",
        "early_stop",
    ]

    rows = []

    for key in display_keys:
        val = config.get(key, "N/A")
        if key == "task_type":
            val = val.title() if isinstance(val, str) else val
        if key == "batch_size":
            if val is not None:
                val = int(val)
            else:
                if training_progress:
                    val = "Auto-selected batch size by Ludwig:<br>"
                    resolved_val = training_progress.get("batch_size")
                    val += f"<span style='font-size: 0.85em;'>{resolved_val}</span><br>"
                else:
                    val = "auto"
        if key == "learning_rate":
            resolved_val = None
            if val is None or val == "auto":
                if training_progress:
                    resolved_val = training_progress.get("learning_rate")
                    val = (
                        "Auto-selected learning rate by Ludwig:<br>"
                        f"<span style='font-size: 0.85em;'>"
                        f"{resolved_val if resolved_val else val}</span><br>"
                        "<span style='font-size: 0.85em;'>"
                        "Based on model architecture and training setup "
                        "(e.g., fine-tuning).<br>"
                        "See <a href='https://ludwig.ai/latest/configuration/trainer/"
                        "#trainer-parameters' target='_blank'>"
                        "Ludwig Trainer Parameters</a> for details."
                        "</span>"
                    )
                else:
                    val = (
                        "Auto-selected by Ludwig<br>"
                        "<span style='font-size: 0.85em;'>"
                        "Automatically tuned based on architecture and dataset.<br>"
                        "See <a href='https://ludwig.ai/latest/configuration/trainer/"
                        "#trainer-parameters' target='_blank'>"
                        "Ludwig Trainer Parameters</a> for details."
                        "</span>"
                    )
            else:
                val = f"{val:.6f}"
        if key == "epochs":
            if (
                training_progress
                and "epoch" in training_progress
                and val > training_progress["epoch"]
            ):
                val = (
                    f"Because of early stopping: the training "
                    f"stopped at epoch {training_progress['epoch']}"
                )

        if val is None:
            continue
        rows.append(
            f"<tr>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left;'>"
            f"{key.replace('_', ' ').title()}</td>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>"
            f"{val}</td>"
            f"</tr>"
        )

    aug_cfg = config.get("augmentation")
    if aug_cfg:
        types = [str(a.get("type", "")) for a in aug_cfg]
        aug_val = ", ".join(types)
        rows.append(
            "<tr>"
            "<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left;'>Augmentation</td>"
            "<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>"
            f"{aug_val}</td>"
            "</tr>"
        )

    if split_info:
        rows.append(
            f"<tr>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left;'>"
            f"Data Split</td>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>"
            f"{split_info}</td>"
            f"</tr>"
        )

    return (
        "<h2 style='text-align: center;'>Training Setup</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; width: 60%; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left;'>"
        "Parameter</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center;'>"
        "Value</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div><br>"
        "<p style='text-align: center; font-size: 0.9em;'>"
        "Model trained using Ludwig.<br>"
        "If want to learn more about Ludwig default settings,"
        "please check their <a href='https://ludwig.ai' target='_blank'>"
        "website(ludwig.ai)</a>."
        "</p><hr>"
    )


def detect_output_type(test_stats):
    """Detects if the output type is 'binary' or 'category' based on test statistics."""
    label_stats = test_stats.get("label", {})
    if "mean_squared_error" in label_stats:
        return "regression"
    per_class = label_stats.get("per_class_stats", {})
    if len(per_class) == 2:
        return "binary"
    return "category"


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
            logging.warning(f"No statistics found for {split} split")
            continue
        label_stats = split_stats.get("label", {})
        if not label_stats:
            logging.warning(f"No label statistics found for {split} split")
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
        logging.warning("No label statistics found for test split")
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


def generate_table_row(cells, styles):
    """Helper function to generate an HTML table row."""
    return (
        "<tr>"
        + "".join(f"<td style='{styles}'>{cell}</td>" for cell in cells)
        + "</tr>"
    )


def format_stats_table_html(train_stats: dict, test_stats: dict) -> str:
    """Formats a combined HTML table for training, validation, and test metrics."""
    output_type = detect_output_type(test_stats)
    all_metrics = extract_metrics_from_json(train_stats, test_stats, output_type)
    rows = []
    for metric_key in sorted(all_metrics["training"].keys()):
        if (
            metric_key in all_metrics["validation"]
            and metric_key in all_metrics["test"]
        ):
            display_name = METRIC_DISPLAY_NAMES.get(
                metric_key,
                metric_key.replace("_", " ").title(),
            )
            t = all_metrics["training"].get(metric_key)
            v = all_metrics["validation"].get(metric_key)
            te = all_metrics["test"].get(metric_key)
            if all(x is not None for x in [t, v, te]):
                rows.append([display_name, f"{t:.4f}", f"{v:.4f}", f"{te:.4f}"])

    if not rows:
        return "<table><tr><td>No metric values found.</td></tr></table>"

    html = (
        "<h2 style='text-align: center;'>Model Performance Summary</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left; "
        "white-space: nowrap;'>Metric</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; "
        "white-space: nowrap;'>Train</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; "
        "white-space: nowrap;'>Validation</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; "
        "white-space: nowrap;'>Test</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; "
            "white-space: nowrap;",
        )
    html += "</tbody></table></div><br>"
    return html


def format_train_val_stats_table_html(train_stats: dict, test_stats: dict) -> str:
    """Formats an HTML table for training and validation metrics."""
    output_type = detect_output_type(test_stats)
    all_metrics = extract_metrics_from_json(train_stats, test_stats, output_type)
    rows = []
    for metric_key in sorted(all_metrics["training"].keys()):
        if metric_key in all_metrics["validation"]:
            display_name = METRIC_DISPLAY_NAMES.get(
                metric_key,
                metric_key.replace("_", " ").title(),
            )
            t = all_metrics["training"].get(metric_key)
            v = all_metrics["validation"].get(metric_key)
            if t is not None and v is not None:
                rows.append([display_name, f"{t:.4f}", f"{v:.4f}"])

    if not rows:
        return "<table><tr><td>No metric values found for Train/Validation.</td></tr></table>"

    html = (
        "<h2 style='text-align: center;'>Train/Validation Performance Summary</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left; "
        "white-space: nowrap;'>Metric</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; "
        "white-space: nowrap;'>Train</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; "
        "white-space: nowrap;'>Validation</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; "
            "white-space: nowrap;",
        )
    html += "</tbody></table></div><br>"
    return html


def format_test_merged_stats_table_html(
    test_metrics: Dict[str, Optional[float]],
) -> str:
    """Formats an HTML table for test metrics."""
    rows = []
    for key in sorted(test_metrics.keys()):
        display_name = METRIC_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
        value = test_metrics[key]
        if value is not None:
            rows.append([display_name, f"{value:.4f}"])

    if not rows:
        return "<table><tr><td>No test metric values found.</td></tr></table>"

    html = (
        "<h2 style='text-align: center;'>Test Performance Summary</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left; "
        "white-space: nowrap;'>Metric</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; "
        "white-space: nowrap;'>Test</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; "
            "white-space: nowrap;",
        )
    html += "</tbody></table></div><br>"
    return html


def split_data_0_2(
    df: pd.DataFrame,
    split_column: str,
    validation_size: float = 0.15,
    random_state: int = 42,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Given a DataFrame whose split_column only contains {0,2}, re-assign a portion of the 0s to become 1s (validation)."""
    out = df.copy()
    out[split_column] = pd.to_numeric(out[split_column], errors="coerce").astype(int)

    idx_train = out.index[out[split_column] == 0].tolist()

    if not idx_train:
        logger.info("No rows with split=0; nothing to do.")
        return out
    stratify_arr = None
    if label_column and label_column in out.columns:
        label_counts = out.loc[idx_train, label_column].value_counts()
        if label_counts.size > 1 and (label_counts.min() * validation_size) >= 1:
            stratify_arr = out.loc[idx_train, label_column]
        else:
            logger.warning(
                "Cannot stratify (too few labels); splitting without stratify."
            )
    if validation_size <= 0:
        logger.info("validation_size <= 0; keeping all as train.")
        return out
    if validation_size >= 1:
        logger.info("validation_size >= 1; moving all train → validation.")
        out.loc[idx_train, split_column] = 1
        return out
    try:
        train_idx, val_idx = train_test_split(
            idx_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify_arr,
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}); retrying without stratify.")
        train_idx, val_idx = train_test_split(
            idx_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=None,
        )
    out.loc[train_idx, split_column] = 0
    out.loc[val_idx, split_column] = 1
    out[split_column] = out[split_column].astype(int)
    return out


class Backend(Protocol):
    """Interface for a machine learning backend."""

    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any],
    ) -> str:
        ...

    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
        random_seed: int,
    ) -> None:
        ...

    def generate_plots(self, output_dir: Path) -> None:
        ...

    def generate_html_report(
        self,
        title: str,
        output_dir: str,
        config: Dict[str, Any],
        split_info: str,
    ) -> Path:
        ...


class LudwigDirectBackend:
    """Backend for running Ludwig experiments directly via the internal experiment_cli function."""

    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any],
    ) -> str:
        logger.info("LudwigDirectBackend: Preparing YAML configuration.")

        model_name = config_params.get("model_name", "resnet18")
        use_pretrained = config_params.get("use_pretrained", False)
        fine_tune = config_params.get("fine_tune", False)
        if use_pretrained:
            trainable = bool(fine_tune)
        else:
            trainable = True
        epochs = config_params.get("epochs", 10)
        batch_size = config_params.get("batch_size")
        num_processes = config_params.get("preprocessing_num_processes", 1)
        early_stop = config_params.get("early_stop", None)
        learning_rate = config_params.get("learning_rate")
        learning_rate = "auto" if learning_rate is None else float(learning_rate)
        raw_encoder = MODEL_ENCODER_TEMPLATES.get(model_name, model_name)
        if isinstance(raw_encoder, dict):
            encoder_config = {
                **raw_encoder,
                "use_pretrained": use_pretrained,
                "trainable": trainable,
            }
        else:
            encoder_config = {"type": raw_encoder}

        batch_size_cfg = batch_size or "auto"

        label_column_path = config_params.get("label_column_data_path")
        label_series = None
        if label_column_path is not None and Path(label_column_path).exists():
            try:
                label_series = pd.read_csv(label_column_path)[LABEL_COLUMN_NAME]
            except Exception as e:
                logger.warning(f"Could not read label column for task detection: {e}")

        if (
            label_series is not None
            and ptypes.is_numeric_dtype(label_series.dtype)
            and label_series.nunique() > 10
        ):
            task_type = "regression"
        else:
            task_type = "classification"

        config_params["task_type"] = task_type

        image_feat: Dict[str, Any] = {
            "name": IMAGE_PATH_COLUMN_NAME,
            "type": "image",
            "encoder": encoder_config,
        }
        if config_params.get("augmentation") is not None:
            image_feat["augmentation"] = config_params["augmentation"]

        if task_type == "regression":
            output_feat = {
                "name": LABEL_COLUMN_NAME,
                "type": "number",
                "decoder": {"type": "regressor"},
                "loss": {"type": "mean_squared_error"},
                "evaluation": {
                    "metrics": [
                        "mean_squared_error",
                        "mean_absolute_error",
                        "r2",
                    ]
                },
            }
            val_metric = config_params.get("validation_metric", "mean_squared_error")

        else:
            num_unique_labels = (
                label_series.nunique() if label_series is not None else 2
            )
            output_type = "binary" if num_unique_labels == 2 else "category"
            output_feat = {"name": LABEL_COLUMN_NAME, "type": output_type}
            val_metric = None

        conf: Dict[str, Any] = {
            "model_type": "ecd",
            "input_features": [image_feat],
            "output_features": [output_feat],
            "combiner": {"type": "concat"},
            "trainer": {
                "epochs": epochs,
                "early_stop": early_stop,
                "batch_size": batch_size_cfg,
                "learning_rate": learning_rate,
                # only set validation_metric for regression
                **({"validation_metric": val_metric} if val_metric else {}),
            },
            "preprocessing": {
                "split": split_config,
                "num_processes": num_processes,
                "in_memory": False,
            },
        }

        logger.debug("LudwigDirectBackend: Config dict built.")
        try:
            yaml_str = yaml.dump(conf, sort_keys=False, indent=2)
            logger.info("LudwigDirectBackend: YAML config generated.")
            return yaml_str
        except Exception:
            logger.error(
                "LudwigDirectBackend: Failed to serialize YAML.",
                exc_info=True,
            )
            raise

    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
        random_seed: int = 42,
    ) -> None:
        """Invoke Ludwig's internal experiment_cli function to run the experiment."""
        logger.info("LudwigDirectBackend: Starting experiment execution.")

        try:
            from ludwig.experiment import experiment_cli
        except ImportError as e:
            logger.error(
                "LudwigDirectBackend: Could not import experiment_cli.",
                exc_info=True,
            )
            raise RuntimeError("Ludwig import failed.") from e

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiment_cli(
                dataset=str(dataset_path),
                config=str(config_path),
                output_directory=str(output_dir),
                random_seed=random_seed,
            )
            logger.info(
                f"LudwigDirectBackend: Experiment completed. Results in {output_dir}"
            )
        except TypeError as e:
            logger.error(
                "LudwigDirectBackend: Argument mismatch in experiment_cli call.",
                exc_info=True,
            )
            raise RuntimeError("Ludwig argument error.") from e
        except Exception:
            logger.error(
                "LudwigDirectBackend: Experiment execution error.",
                exc_info=True,
            )
            raise

    def get_training_process(self, output_dir) -> Optional[Dict[str, Any]]:
        """Retrieve the learning rate used in the most recent Ludwig run."""
        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )

        if not exp_dirs:
            logger.warning(f"No experiment run directories found in {output_dir}")
            return None

        progress_file = exp_dirs[-1] / "model" / "training_progress.json"
        if not progress_file.exists():
            logger.warning(f"No training_progress.json found in {progress_file}")
            return None

        try:
            with progress_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "learning_rate": data.get("learning_rate"),
                "batch_size": data.get("batch_size"),
                "epoch": data.get("epoch"),
            }
        except Exception as e:
            logger.warning(f"Failed to read training progress info: {e}")
            return {}

    def convert_parquet_to_csv(self, output_dir: Path):
        """Convert the predictions Parquet file to CSV."""
        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not exp_dirs:
            logger.warning(f"No experiment run dirs found in {output_dir}")
            return
        exp_dir = exp_dirs[-1]
        parquet_path = exp_dir / PREDICTIONS_PARQUET_FILE_NAME
        csv_path = exp_dir / "predictions.csv"
        try:
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
            logger.info(f"Converted Parquet to CSV: {csv_path}")
        except Exception as e:
            logger.error(f"Error converting Parquet to CSV: {e}")

    def generate_plots(self, output_dir: Path) -> None:
        """Generate all registered Ludwig visualizations for the latest experiment run."""
        logger.info("Generating all Ludwig visualizations…")

        test_plots = {
            "compare_performance",
            "compare_classifiers_performance_from_prob",
            "compare_classifiers_performance_from_pred",
            "compare_classifiers_performance_changing_k",
            "compare_classifiers_multiclass_multimetric",
            "compare_classifiers_predictions",
            "confidence_thresholding_2thresholds_2d",
            "confidence_thresholding_2thresholds_3d",
            "confidence_thresholding",
            "confidence_thresholding_data_vs_acc",
            "binary_threshold_vs_metric",
            "roc_curves",
            "roc_curves_from_test_statistics",
            "calibration_1_vs_all",
            "calibration_multiclass",
            "confusion_matrix",
            "frequency_vs_f1",
        }
        train_plots = {
            "learning_curves",
            "compare_classifiers_performance_subset",
        }

        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not exp_dirs:
            logger.warning(f"No experiment run dirs found in {output_dir}")
            return
        exp_dir = exp_dirs[-1]

        viz_dir = exp_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        train_viz = viz_dir / "train"
        test_viz = viz_dir / "test"
        train_viz.mkdir(parents=True, exist_ok=True)
        test_viz.mkdir(parents=True, exist_ok=True)

        def _check(p: Path) -> Optional[str]:
            return str(p) if p.exists() else None

        training_stats = _check(exp_dir / "training_statistics.json")
        test_stats = _check(exp_dir / TEST_STATISTICS_FILE_NAME)
        probs_path = _check(exp_dir / PREDICTIONS_PARQUET_FILE_NAME)
        gt_metadata = _check(exp_dir / "model" / TRAIN_SET_METADATA_FILE_NAME)

        dataset_path = None
        split_file = None
        desc = exp_dir / DESCRIPTION_FILE_NAME
        if desc.exists():
            with open(desc, "r") as f:
                cfg = json.load(f)
            dataset_path = _check(Path(cfg.get("dataset", "")))
            split_file = _check(Path(get_split_path(cfg.get("dataset", ""))))

        output_feature = ""
        if desc.exists():
            try:
                output_feature = cfg["config"]["output_features"][0]["name"]
            except Exception:
                pass
        if not output_feature and test_stats:
            with open(test_stats, "r") as f:
                stats = json.load(f)
            output_feature = next(iter(stats.keys()), "")

        viz_registry = get_visualizations_registry()
        for viz_name, viz_func in viz_registry.items():
            if viz_name in train_plots:
                viz_dir_plot = train_viz
            elif viz_name in test_plots:
                viz_dir_plot = test_viz
            else:
                continue

            try:
                viz_func(
                    training_statistics=[training_stats] if training_stats else [],
                    test_statistics=[test_stats] if test_stats else [],
                    probabilities=[probs_path] if probs_path else [],
                    output_feature_name=output_feature,
                    ground_truth_split=2,
                    top_n_classes=[0],
                    top_k=3,
                    ground_truth_metadata=gt_metadata,
                    ground_truth=dataset_path,
                    split_file=split_file,
                    output_directory=str(viz_dir_plot),
                    normalize=False,
                    file_format="png",
                )
                logger.info(f"✔ Generated {viz_name}")
            except Exception as e:
                logger.warning(f"✘ Skipped {viz_name}: {e}")

        logger.info(f"All visualizations written to {viz_dir}")

    def generate_html_report(
        self,
        title: str,
        output_dir: str,
        config: dict,
        split_info: str,
    ) -> Path:
        """Assemble an HTML report from visualizations under train_val/ and test/ folders."""
        cwd = Path.cwd()
        report_name = title.lower().replace(" ", "_") + "_report.html"
        report_path = cwd / report_name
        output_dir = Path(output_dir)
        output_type = None

        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not exp_dirs:
            raise RuntimeError(f"No 'experiment*' dirs found in {output_dir}")
        exp_dir = exp_dirs[-1]

        base_viz_dir = exp_dir / "visualizations"
        train_viz_dir = base_viz_dir / "train"
        test_viz_dir = base_viz_dir / "test"

        html = get_html_template()
        html += f"<h1>{title}</h1>"

        metrics_html = ""
        train_val_metrics_html = ""
        test_metrics_html = ""
        try:
            train_stats_path = exp_dir / "training_statistics.json"
            test_stats_path = exp_dir / TEST_STATISTICS_FILE_NAME
            if train_stats_path.exists() and test_stats_path.exists():
                with open(train_stats_path) as f:
                    train_stats = json.load(f)
                with open(test_stats_path) as f:
                    test_stats = json.load(f)
                output_type = detect_output_type(test_stats)
                metrics_html = format_stats_table_html(train_stats, test_stats)
                train_val_metrics_html = format_train_val_stats_table_html(
                    train_stats, test_stats
                )
                test_metrics_html = format_test_merged_stats_table_html(
                    extract_metrics_from_json(train_stats, test_stats, output_type)[
                        "test"
                    ]
                )
        except Exception as e:
            logger.warning(
                f"Could not load stats for HTML report: {type(e).__name__}: {e}"
            )

        config_html = ""
        training_progress = self.get_training_process(output_dir)
        try:
            config_html = format_config_table_html(
                config, split_info, training_progress
            )
        except Exception as e:
            logger.warning(f"Could not load config for HTML report: {e}")

        def render_img_section(
            title: str, dir_path: Path, output_type: str = None
        ) -> str:
            if not dir_path.exists():
                return f"<h2>{title}</h2><p><em>Directory not found.</em></p>"

            imgs = list(dir_path.glob("*.png"))
            if not imgs:
                return f"<h2>{title}</h2><p><em>No plots found.</em></p>"

            if title == "Test Visualizations" and output_type == "binary":
                order = [
                    "confusion_matrix__label_top2.png",
                    "roc_curves_from_prediction_statistics.png",
                    "compare_performance_label.png",
                    "confusion_matrix_entropy__label_top2.png",
                ]
                img_names = {img.name: img for img in imgs}
                ordered_imgs = [
                    img_names[fname] for fname in order if fname in img_names
                ]
                remaining = sorted(
                    [
                        img
                        for img in imgs
                        if img.name not in order and img.name != "roc_curves.png"
                    ]
                )
                imgs = ordered_imgs + remaining

            elif title == "Test Visualizations" and output_type == "category":
                unwanted = {
                    "compare_classifiers_multiclass_multimetric__label_best10.png",
                    "compare_classifiers_multiclass_multimetric__label_top10.png",
                    "compare_classifiers_multiclass_multimetric__label_worst10.png",
                }
                display_order = [
                    "confusion_matrix__label_top10.png",
                    "roc_curves.png",
                    "compare_performance_label.png",
                    "compare_classifiers_performance_from_prob.png",
                    "compare_classifiers_multiclass_multimetric__label_sorted.png",
                    "confusion_matrix_entropy__label_top10.png",
                ]
                img_names = {img.name: img for img in imgs if img.name not in unwanted}
                ordered_imgs = [
                    img_names[fname] for fname in display_order if fname in img_names
                ]
                remaining = sorted(
                    [img for img in img_names.values() if img.name not in display_order]
                )
                imgs = ordered_imgs + remaining

            else:
                if output_type == "category":
                    unwanted = {
                        "compare_classifiers_multiclass_multimetric__label_best10.png",
                        "compare_classifiers_multiclass_multimetric__label_top10.png",
                        "compare_classifiers_multiclass_multimetric__label_worst10.png",
                    }
                    imgs = sorted([img for img in imgs if img.name not in unwanted])
                else:
                    imgs = sorted(imgs)

            section_html = f"<h2 style='text-align: center;'>{title}</h2><div>"
            for img in imgs:
                b64 = encode_image_to_base64(str(img))
                section_html += (
                    f'<div class="plot" style="margin-bottom:20px;text-align:center;">'
                    f"<h3>{img.stem.replace('_', ' ').title()}</h3>"
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:90%;max-height:600px;border:1px solid #ddd;" />'
                    f"</div>"
                )
            section_html += "</div>"
            return section_html

        tab1_content = config_html + metrics_html

        tab2_content = train_val_metrics_html + render_img_section(
            "Training & Validation Visualizations", train_viz_dir
        )

        # --- Predictions vs Ground Truth table ---
        preds_section = ""
        parquet_path = exp_dir / PREDICTIONS_PARQUET_FILE_NAME
        if parquet_path.exists():
            try:
                # 1) load predictions from Parquet
                df_preds = pd.read_parquet(parquet_path).reset_index(drop=True)
                # assume the column containing your model's prediction is named "prediction"
                # or contains that substring:
                pred_col = next(
                    (c for c in df_preds.columns if "prediction" in c.lower()),
                    None,
                )
                if pred_col is None:
                    raise ValueError("No prediction column found in Parquet output")
                df_pred = df_preds[[pred_col]].rename(columns={pred_col: "prediction"})

                # 2) load ground truth for the test split from prepared CSV
                df_all = pd.read_csv(config["label_column_data_path"])
                df_gt = df_all[df_all[SPLIT_COLUMN_NAME] == 2][
                    LABEL_COLUMN_NAME
                ].reset_index(drop=True)

                # 3) concatenate side‐by‐side
                df_table = pd.concat([df_gt, df_pred], axis=1)
                df_table.columns = [LABEL_COLUMN_NAME, "prediction"]

                # 4) render as HTML
                preds_html = df_table.to_html(index=False, classes="predictions-table")
                preds_section = (
                    "<h2 style='text-align: center;'>Predictions vs. Ground Truth</h2>"
                    "<div style='overflow-x:auto; margin-bottom:20px;'>"
                    + preds_html
                    + "</div>"
                )
            except Exception as e:
                logger.warning(f"Could not build Predictions vs GT table: {e}")
        # Test tab = Metrics + Preds table + Visualizations

        tab3_content = (
            test_metrics_html
            + preds_section
            + render_img_section("Test Visualizations", test_viz_dir, output_type)
        )

        # assemble the tabs and help modal
        tabbed_html = build_tabbed_html(tab1_content, tab2_content, tab3_content)
        modal_html = get_metrics_help_modal()
        html += tabbed_html + modal_html + get_html_closing()

        try:
            with open(report_path, "w") as f:
                f.write(html)
            logger.info(f"HTML report generated at: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            raise

        return report_path


class WorkflowOrchestrator:
    """Manages the image-classification workflow."""

    def __init__(self, args: argparse.Namespace, backend: Backend):
        self.args = args
        self.backend = backend
        self.temp_dir: Optional[Path] = None
        self.image_extract_dir: Optional[Path] = None
        logger.info(f"Orchestrator initialized with backend: {type(backend).__name__}")

    def _create_temp_dirs(self) -> None:
        """Create temporary output and image extraction directories."""
        try:
            self.temp_dir = Path(
                tempfile.mkdtemp(dir=self.args.output_dir, prefix=TEMP_DIR_PREFIX)
            )
            self.image_extract_dir = self.temp_dir / "images"
            self.image_extract_dir.mkdir()
            logger.info(f"Created temp directory: {self.temp_dir}")
        except Exception:
            logger.error("Failed to create temporary directories", exc_info=True)
            raise

    def _extract_images(self) -> None:
        """Extract images from ZIP into the temp image directory."""
        if self.image_extract_dir is None:
            raise RuntimeError("Temp image directory not initialized.")
        logger.info(
            f"Extracting images from {self.args.image_zip} → {self.image_extract_dir}"
        )
        try:
            with zipfile.ZipFile(self.args.image_zip, "r") as z:
                z.extractall(self.image_extract_dir)
            logger.info("Image extraction complete.")
        except Exception:
            logger.error("Error extracting zip file", exc_info=True)
            raise

    def _prepare_data(self) -> Tuple[Path, Dict[str, Any], str]:
        """Load CSV, update image paths, handle splits, and write prepared CSV."""
        if not self.temp_dir or not self.image_extract_dir:
            raise RuntimeError("Temp dirs not initialized before data prep.")

        try:
            df = pd.read_csv(self.args.csv_file)
            logger.info(f"Loaded CSV: {self.args.csv_file}")
        except Exception:
            logger.error("Error loading CSV file", exc_info=True)
            raise

        required = {IMAGE_PATH_COLUMN_NAME, LABEL_COLUMN_NAME}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        try:
            df[IMAGE_PATH_COLUMN_NAME] = df[IMAGE_PATH_COLUMN_NAME].apply(
                lambda p: str((self.image_extract_dir / p).resolve())
            )
        except Exception:
            logger.error("Error updating image paths", exc_info=True)
            raise

        if SPLIT_COLUMN_NAME in df.columns:
            df, split_config, split_info = self._process_fixed_split(df)
        else:
            logger.info("No split column; using random split")
            split_config = {
                "type": "random",
                "probabilities": self.args.split_probabilities,
            }
            split_info = (
                f"No split column in CSV. Used random split: "
                f"{[int(p * 100) for p in self.args.split_probabilities]}% "
                f"for train/val/test."
            )

        final_csv = self.temp_dir / TEMP_CSV_FILENAME
        try:

            df.to_csv(final_csv, index=False)
            logger.info(f"Saved prepared data to {final_csv}")
        except Exception:
            logger.error("Error saving prepared CSV", exc_info=True)
            raise

        return final_csv, split_config, split_info

    def _process_fixed_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """Process a fixed split column (0=train,1=val,2=test)."""
        logger.info(f"Fixed split column '{SPLIT_COLUMN_NAME}' detected.")
        try:
            col = df[SPLIT_COLUMN_NAME]
            df[SPLIT_COLUMN_NAME] = pd.to_numeric(col, errors="coerce").astype(
                pd.Int64Dtype()
            )
            if df[SPLIT_COLUMN_NAME].isna().any():
                logger.warning("Split column contains non-numeric/missing values.")

            unique = set(df[SPLIT_COLUMN_NAME].dropna().unique())
            logger.info(f"Unique split values: {unique}")

            if unique == {0, 2}:
                df = split_data_0_2(
                    df,
                    SPLIT_COLUMN_NAME,
                    validation_size=self.args.validation_size,
                    label_column=LABEL_COLUMN_NAME,
                    random_state=self.args.random_seed,
                )
                split_info = (
                    "Detected a split column (with values 0 and 2) in the input CSV. "
                    f"Used this column as a base and reassigned "
                    f"{self.args.validation_size * 100:.1f}% "
                    "of the training set (originally labeled 0) to validation (labeled 1)."
                )
                logger.info("Applied custom 0/2 split.")
            elif unique.issubset({0, 1, 2}):
                split_info = "Used user-defined split column from CSV."
                logger.info("Using fixed split as-is.")
            else:
                raise ValueError(f"Unexpected split values: {unique}")

            return df, {"type": "fixed", "column": SPLIT_COLUMN_NAME}, split_info

        except Exception:
            logger.error("Error processing fixed split", exc_info=True)
            raise

    def _cleanup_temp_dirs(self) -> None:
        if self.temp_dir and self.temp_dir.exists():
            logger.info(f"Cleaning up temp directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.image_extract_dir = None

    def run(self) -> None:
        """Execute the full workflow end-to-end."""
        logger.info("Starting workflow...")
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._create_temp_dirs()
            self._extract_images()
            csv_path, split_cfg, split_info = self._prepare_data()

            use_pretrained = self.args.use_pretrained or self.args.fine_tune

            backend_args = {
                "model_name": self.args.model_name,
                "fine_tune": self.args.fine_tune,
                "use_pretrained": use_pretrained,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "preprocessing_num_processes": self.args.preprocessing_num_processes,
                "split_probabilities": self.args.split_probabilities,
                "learning_rate": self.args.learning_rate,
                "random_seed": self.args.random_seed,
                "early_stop": self.args.early_stop,
                "label_column_data_path": csv_path,
                "augmentation": self.args.augmentation,
            }
            yaml_str = self.backend.prepare_config(backend_args, split_cfg)

            config_file = self.temp_dir / TEMP_CONFIG_FILENAME
            config_file.write_text(yaml_str)
            logger.info(f"Wrote backend config: {config_file}")

            self.backend.run_experiment(
                csv_path,
                config_file,
                self.args.output_dir,
                self.args.random_seed,
            )
            logger.info("Workflow completed successfully.")
            self.backend.generate_plots(self.args.output_dir)
            report_file = self.backend.generate_html_report(
                "Image Classification Results",
                self.args.output_dir,
                backend_args,
                split_info,
            )
            logger.info(f"HTML report generated at: {report_file}")
            self.backend.convert_parquet_to_csv(self.args.output_dir)
            logger.info("Converted Parquet to CSV.")
        except Exception:
            logger.error("Workflow execution failed", exc_info=True)
            raise
        finally:
            self._cleanup_temp_dirs()


def parse_learning_rate(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


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
        if key not in mapping:
            valid = ", ".join(mapping.keys())
            raise ValueError(f"Unknown augmentation '{key}'. Valid choices: {valid}")
        aug_list.append(mapping[key])
    return aug_list


class SplitProbAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        train, val, test = values
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            parser.error(
                f"--split-probabilities must sum to 1.0; "
                f"got {train:.3f} + {val:.3f} + {test:.3f} = {total:.3f}"
            )
        setattr(namespace, self.dest, values)


def main():
    parser = argparse.ArgumentParser(
        description="Image Classification Learner with Pluggable Backends",
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        type=Path,
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--image-zip",
        required=True,
        type=Path,
        help="Path to the images ZIP",
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
            "Random split proportions (e.g., 0.7 0.1 0.2)."
            "Only used if no split column."
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

    args = parser.parse_args()

    if not 0.0 <= args.validation_size <= 1.0:
        parser.error("validation-size must be between 0.0 and 1.0")
    if not args.csv_file.is_file():
        parser.error(f"CSV not found: {args.csv_file}")
    if not args.image_zip.is_file():
        parser.error(f"ZIP not found: {args.image_zip}")
    if args.augmentation is not None:
        try:
            augmentation_setup = aug_parse(args.augmentation)
            setattr(args, "augmentation", augmentation_setup)
        except ValueError as e:
            parser.error(str(e))

    backend_instance = LudwigDirectBackend()
    orchestrator = WorkflowOrchestrator(args, backend_instance)

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
