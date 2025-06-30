#!/usr/bin/env python3
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
import yaml
from ludwig.globals import (
    DESCRIPTION_FILE_NAME,
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME,
)
from ludwig.utils.data_utils import get_split_path
from ludwig.visualize import get_visualizations_registry
from sklearn.model_selection import train_test_split
from utils import encode_image_to_base64, get_html_closing, get_html_template

# --- Constants ---
SPLIT_COLUMN_NAME = "split"
LABEL_COLUMN_NAME = "label"
IMAGE_PATH_COLUMN_NAME = "image_path"
DEFAULT_SPLIT_PROBABILITIES = [0.7, 0.1, 0.2]
TEMP_CSV_FILENAME = "processed_data_for_ludwig.csv"
TEMP_CONFIG_FILENAME = "ludwig_config.yaml"
TEMP_DIR_PREFIX = "ludwig_api_work_"
MODEL_ENCODER_TEMPLATES: Dict[str, Any] = {
    "stacked_cnn": "stacked_cnn",
    "resnet18": {"type": "resnet", "model_variant": 18},
    "resnet34": {"type": "resnet", "model_variant": 34},
    "resnet50": {"type": "resnet", "model_variant": 50},
    "resnet101": {"type": "resnet", "model_variant": 101},
    "resnet152": {"type": "resnet", "model_variant": 152},
    "resnext50_32x4d": {"type": "resnext", "model_variant": "50_32x4d"},
    "resnext101_32x8d": {"type": "resnext", "model_variant": "101_32x8d"},
    "resnext101_64x4d": {"type": "resnext", "model_variant": "101_64x4d"},
    "resnext152_32x8d": {"type": "resnext", "model_variant": "152_32x8d"},
    "wide_resnet50_2": {"type": "wide_resnet", "model_variant": "50_2"},
    "wide_resnet101_2": {"type": "wide_resnet", "model_variant": "101_2"},
    "wide_resnet103_2": {"type": "wide_resnet", "model_variant": "103_2"},
    "efficientnet_b0": {"type": "efficientnet", "model_variant": "b0"},
    "efficientnet_b1": {"type": "efficientnet", "model_variant": "b1"},
    "efficientnet_b2": {"type": "efficientnet", "model_variant": "b2"},
    "efficientnet_b3": {"type": "efficientnet", "model_variant": "b3"},
    "efficientnet_b4": {"type": "efficientnet", "model_variant": "b4"},
    "efficientnet_b5": {"type": "efficientnet", "model_variant": "b5"},
    "efficientnet_b6": {"type": "efficientnet", "model_variant": "b6"},
    "efficientnet_b7": {"type": "efficientnet", "model_variant": "b7"},
    "efficientnet_v2_s": {"type": "efficientnet", "model_variant": "v2_s"},
    "efficientnet_v2_m": {"type": "efficientnet", "model_variant": "v2_m"},
    "efficientnet_v2_l": {"type": "efficientnet", "model_variant": "v2_l"},
    "regnet_y_400mf": {"type": "regnet", "model_variant": "y_400mf"},
    "regnet_y_800mf": {"type": "regnet", "model_variant": "y_800mf"},
    "regnet_y_1_6gf": {"type": "regnet", "model_variant": "y_1_6gf"},
    "regnet_y_3_2gf": {"type": "regnet", "model_variant": "y_3_2gf"},
    "regnet_y_8gf": {"type": "regnet", "model_variant": "y_8gf"},
    "regnet_y_16gf": {"type": "regnet", "model_variant": "y_16gf"},
    "regnet_y_32gf": {"type": "regnet", "model_variant": "y_32gf"},
    "regnet_y_128gf": {"type": "regnet", "model_variant": "y_128gf"},
    "regnet_x_400mf": {"type": "regnet", "model_variant": "x_400mf"},
    "regnet_x_800mf": {"type": "regnet", "model_variant": "x_800mf"},
    "regnet_x_1_6gf": {"type": "regnet", "model_variant": "x_1_6gf"},
    "regnet_x_3_2gf": {"type": "regnet", "model_variant": "x_3_2gf"},
    "regnet_x_8gf": {"type": "regnet", "model_variant": "x_8gf"},
    "regnet_x_16gf": {"type": "regnet", "model_variant": "x_16gf"},
    "regnet_x_32gf": {"type": "regnet", "model_variant": "x_32gf"},
    "vgg11": {"type": "vgg", "model_variant": 11},
    "vgg11_bn": {"type": "vgg", "model_variant": "11_bn"},
    "vgg13": {"type": "vgg", "model_variant": 13},
    "vgg13_bn": {"type": "vgg", "model_variant": "13_bn"},
    "vgg16": {"type": "vgg", "model_variant": 16},
    "vgg16_bn": {"type": "vgg", "model_variant": "16_bn"},
    "vgg19": {"type": "vgg", "model_variant": 19},
    "vgg19_bn": {"type": "vgg", "model_variant": "19_bn"},
    "shufflenet_v2_x0_5": {"type": "shufflenet_v2", "model_variant": "x0_5"},
    "shufflenet_v2_x1_0": {"type": "shufflenet_v2", "model_variant": "x1_0"},
    "shufflenet_v2_x1_5": {"type": "shufflenet_v2", "model_variant": "x1_5"},
    "shufflenet_v2_x2_0": {"type": "shufflenet_v2", "model_variant": "x2_0"},
    "squeezenet1_0": {"type": "squeezenet", "model_variant": "1_0"},
    "squeezenet1_1": {"type": "squeezenet", "model_variant": "1_1"},
    "swin_t": {"type": "swin_transformer", "model_variant": "t"},
    "swin_s": {"type": "swin_transformer", "model_variant": "s"},
    "swin_b": {"type": "swin_transformer", "model_variant": "b"},
    "swin_v2_t": {"type": "swin_transformer", "model_variant": "v2_t"},
    "swin_v2_s": {"type": "swin_transformer", "model_variant": "v2_s"},
    "swin_v2_b": {"type": "swin_transformer", "model_variant": "v2_b"},
    "vit_b_16": {"type": "vision_transformer", "model_variant": "b_16"},
    "vit_b_32": {"type": "vision_transformer", "model_variant": "b_32"},
    "vit_l_16": {"type": "vision_transformer", "model_variant": "l_16"},
    "vit_l_32": {"type": "vision_transformer", "model_variant": "l_32"},
    "vit_h_14": {"type": "vision_transformer", "model_variant": "h_14"},
    "convnext_tiny": {"type": "convnext", "model_variant": "tiny"},
    "convnext_small": {"type": "convnext", "model_variant": "small"},
    "convnext_base": {"type": "convnext", "model_variant": "base"},
    "convnext_large": {"type": "convnext", "model_variant": "large"},
    "maxvit_t": {"type": "maxvit", "model_variant": "t"},
    "alexnet": {"type": "alexnet"},
    "googlenet": {"type": "googlenet"},
    "inception_v3": {"type": "inception_v3"},
    "mobilenet_v2": {"type": "mobilenet_v2"},
    "mobilenet_v3_large": {"type": "mobilenet_v3_large"},
    "mobilenet_v3_small": {"type": "mobilenet_v3_small"},
}
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "accuracy_micro": "Accuracy-Micro",
    "loss": "Loss",
    "roc_auc": "ROC-AUC",
    "roc_auc_macro": "ROC-AUC-Macro",
    "roc_auc_micro": "ROC-AUC-Micro",
    "hits_at_k": "Hits at K",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "kappa_score": "Cohen's Kappa",
    "token_accuracy": "Token Accuracy",
    "avg_precision_macro": "Precision-Macro",
    "avg_recall_macro": "Recall-Macro",
    "avg_f1_score_macro": "F1-score-Macro",
    "avg_precision_micro": "Precision-Micro",
    "avg_recall_micro": "Recall-Micro",
    "avg_f1_score_micro": "F1-score-Micro",
    "avg_precision_weighted": "Precision-Weighted",
    "avg_recall_weighted": "Recall-Weighted",
    "avg_f1_score_weighted": "F1-score-Weighted",
    "average_precision_macro": " Precision-Average-Macro",
    "average_precision_micro": "Precision-Average-Micro",
    "average_precision_samples": "Precision-Average-Samples",
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ImageLearner")


def get_metrics_help_modal() -> str:
    modal_html = """
<div id="metricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close">×</span>
    <h2>Model Evaluation Metrics — Help Guide</h2>
    <div class="metrics-guide">
      <h3>1) General Metrics</h3>
      <p><strong>Loss:</strong> Measures the difference between predicted and actual values. Lower is better. Often used for optimization during training.</p>
      <p><strong>Accuracy:</strong> Proportion of correct predictions among all predictions. Simple but can be misleading for imbalanced datasets.</p>
      <p><strong>Micro Accuracy:</strong> Calculates accuracy by summing up all individual true positives and true negatives across all classes, making it suitable for multiclass or multilabel problems.</p>
      <p><strong>Token Accuracy:</strong> Measures how often the predicted tokens (e.g., in sequences) match the true tokens. Useful in sequence prediction tasks like NLP.</p>
      <h3>2) Precision, Recall & Specificity</h3>
      <p><strong>Precision:</strong> Out of all positive predictions, how many were correct. Precision = TP / (TP + FP). Helps when false positives are costly.</p>
      <p><strong>Recall (Sensitivity):</strong> Out of all actual positives, how many were predicted correctly. Recall = TP / (TP + FN). Important when missing positives is risky.</p>
      <p><strong>Specificity:</strong> True negative rate. Measures how well the model identifies negatives. Specificity = TN / (TN + FP). Useful in medical testing to avoid false alarms.</p>
      <h3>3) Macro, Micro, and Weighted Averages</h3>
      <p><strong>Macro Precision / Recall / F1:</strong> Averages the metric across all classes, treating each class equally, regardless of class frequency. Best when class sizes are balanced.</p>
      <p><strong>Micro Precision / Recall / F1:</strong> Aggregates TP, FP, FN across all classes before computing the metric. Gives a global view and is ideal for class-imbalanced problems.</p>
      <p><strong>Weighted Precision / Recall / F1:</strong> Averages each metric across classes, weighted by the number of true instances per class. Balances importance of classes based on frequency.</p>
      <h3>4) Average Precision (PR-AUC Variants)</h3>
      <p><strong>Average Precision Macro:</strong> Precision-Recall AUC averaged across all classes equally. Useful for balanced multi-class problems.</p>
      <p><strong>Average Precision Micro:</strong> Global Precision-Recall AUC using all instances. Best for imbalanced data or multi-label classification.</p>
      <p><strong>Average Precision Samples:</strong> Precision-Recall AUC averaged across individual samples (not classes). Ideal for multi-label problems where each sample can belong to multiple classes.</p>
      <h3>5) ROC-AUC Variants</h3>
      <p><strong>ROC-AUC:</strong> Measures model's ability to distinguish between classes. AUC = 1 is perfect; 0.5 is random guessing. Use for binary classification.</p>
      <p><strong>Macro ROC-AUC:</strong> Averages the AUC across all classes equally. Suitable when classes are balanced and of equal importance.</p>
      <p><strong>Micro ROC-AUC:</strong> Computes AUC from aggregated predictions across all classes. Useful in multiclass or multilabel settings with imbalance.</p>
      <h3>6) Ranking Metrics</h3>
      <p><strong>Hits at K:</strong> Measures whether the true label is among the top-K predictions. Common in recommendation systems and retrieval tasks.</p>
      <h3>7) Confusion Matrix Stats (Per Class)</h3>
      <p><strong>True Positives / Negatives (TP / TN):</strong> Correct predictions for positives and negatives respectively.</p>
      <p><strong>False Positives / Negatives (FP / FN):</strong> Incorrect predictions — false alarms and missed detections.</p>
      <h3>8) Other Useful Metrics</h3>
      <p><strong>Cohen's Kappa:</strong> Measures agreement between predicted and actual values adjusted for chance. Useful for multiclass classification with imbalanced labels.</p>
      <p><strong>Matthews Correlation Coefficient (MCC):</strong> Balanced measure of prediction quality that takes into account TP, TN, FP, and FN. Particularly effective for imbalanced datasets.</p>
      <h3>9) Metric Recommendations</h3>
      <ul>
        <li>Use <strong>Accuracy + F1</strong> for balanced data.</li>
        <li>Use <strong>Precision, Recall, ROC-AUC</strong> for imbalanced datasets.</li>
        <li>Use <strong>Average Precision Micro</strong> for multilabel or class-imbalanced problems.</li>
        <li>Use <strong>Macro scores</strong> when all classes should be treated equally.</li>
        <li>Use <strong>Weighted scores</strong> when class imbalance should be accounted for without ignoring small classes.</li>
        <li>Use <strong>Confusion Matrix stats</strong> to analyze class-wise performance.</li>
        <li>Use <strong>Hits at K</strong> for recommendation or ranking-based tasks.</li>
      </ul>
    </div>
  </div>
</div>
"""
    modal_css = """
<style>
.modal {
  display: none;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.4);
}
.modal-content {
  background-color: #fefefe;
  margin: 15% auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
  max-width: 800px;
}
.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}
.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}
.metrics-guide h3 {
  margin-top: 20px;
}
.metrics-guide p {
  margin: 5px 0;
}
.metrics-guide ul {
  margin: 10px 0;
  padding-left: 20px;
}
</style>
"""
    modal_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  var modal = document.getElementById("metricsHelpModal");
  var closeBtn = document.getElementsByClassName("close")[0];

  document.querySelectorAll(".openMetricsHelp").forEach(btn => {
    btn.onclick = function() {
      modal.style.display = "block";
    };
  });

  if (closeBtn) {
    closeBtn.onclick = function() {
      modal.style.display = "none";
    };
  }

  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
});
</script>
"""
    return modal_css + modal_html + modal_js


def format_config_table_html(
    config: dict,
    split_info: Optional[str] = None,
    training_progress: dict = None,
) -> str:
    display_keys = [
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
        "please check the their <a href='https://ludwig.ai' target='_blank'>"
        "website(ludwig.ai)</a>."
        "</p><hr>"
    )


def detect_output_type(test_stats):
    """Detects if the output type is 'binary' or 'category' based on test statistics."""
    label_stats = test_stats.get("label", {})
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


def format_test_merged_stats_table_html(test_metrics: Dict[str, Optional[float]]) -> str:
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


def build_tabbed_html(metrics_html: str, train_val_html: str, test_html: str) -> str:
    return f"""
<style>
.tabs {{
  display: flex;
  border-bottom: 2px solid #ccc;
  margin-bottom: 1rem;
}}
.tab {{
  padding: 10px 20px;
  cursor: pointer;
  border: 1px solid #ccc;
  border-bottom: none;
  background: #f9f9f9;
  margin-right: 5px;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
}}
.tab.active {{
  background: white;
  font-weight: bold;
}}
.tab-content {{
  display: none;
  padding: 20px;
  border: 1px solid #ccc;
  border-top: none;
}}
.tab-content.active {{
  display: block;
}}
</style>
<div class="tabs">
  <div class="tab active" onclick="showTab('metrics')"> Config & Results Summary</div>
  <div class="tab" onclick="showTab('trainval')"> Train/Validation Results</div>
  <div class="tab" onclick="showTab('test')"> Test Results</div>
</div>
<div id="metrics" class="tab-content active">
  {metrics_html}
</div>
<div id="trainval" class="tab-content">
  {train_val_html}
</div>
<div id="test" class="tab-content">
  {test_html}
</div>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`.tab[onclick*="${{id}}"]`).classList.add('active');
}}
</script>
"""


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

    def generate_plots(
        self,
        output_dir: Path
    ) -> None:
        ...

    def generate_html_report(
        self,
        title: str,
        output_dir: str
    ) -> Path:
        ...


class LudwigDirectBackend:
    """Backend for running Ludwig experiments directly via the internal experiment_cli function."""

    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any],
    ) -> str:
        """Build and serialize the Ludwig YAML configuration."""
        logger.info("LudwigDirectBackend: Preparing YAML configuration.")

        model_name = config_params.get("model_name", "resnet18")
        use_pretrained = config_params.get("use_pretrained", False)
        fine_tune = config_params.get("fine_tune", False)
        epochs = config_params.get("epochs", 10)
        batch_size = config_params.get("batch_size")
        num_processes = config_params.get("preprocessing_num_processes", 1)
        early_stop = config_params.get("early_stop", None)
        learning_rate = config_params.get("learning_rate")
        learning_rate = "auto" if learning_rate is None else float(learning_rate)
        trainable = fine_tune or (not use_pretrained)
        if not use_pretrained and not trainable:
            logger.warning("trainable=False; use_pretrained=False is ignored.")
            logger.warning("Setting trainable=True to train the model from scratch.")
            trainable = True
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
        if label_column_path is not None and Path(label_column_path).exists():
            try:
                label_series = pd.read_csv(label_column_path)[LABEL_COLUMN_NAME]
                num_unique_labels = label_series.nunique()
            except Exception as e:
                logger.warning(
                    f"Could not determine label cardinality, defaulting to 'binary': {e}"
                )
                num_unique_labels = 2
        else:
            logger.warning(
                "label_column_data_path not provided, defaulting to 'binary'"
            )
            num_unique_labels = 2

        output_type = "binary" if num_unique_labels == 2 else "category"

        conf: Dict[str, Any] = {
            "model_type": "ecd",
            "input_features": [
                {
                    "name": IMAGE_PATH_COLUMN_NAME,
                    "type": "image",
                    "encoder": encoder_config,
                }
            ],
            "output_features": [{"name": LABEL_COLUMN_NAME, "type": output_type}],
            "combiner": {"type": "concat"},
            "trainer": {
                "epochs": epochs,
                "early_stop": early_stop,
                "batch_size": batch_size_cfg,
                "learning_rate": learning_rate,
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

    def get_training_process(self, output_dir) -> float:
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
            viz_dir_plot = None
            if viz_name in train_plots:
                viz_dir_plot = train_viz
            elif viz_name in test_plots:
                viz_dir_plot = test_viz

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
                all_metrics = extract_metrics_from_json(
                    train_stats,
                    test_stats,
                    output_type,
                )
                metrics_html = format_stats_table_html(train_stats, test_stats)
                train_val_metrics_html = format_train_val_stats_table_html(
                    train_stats,
                    test_stats,
                )
                test_metrics_html = format_test_merged_stats_table_html(
                    all_metrics["test"],
                )
        except Exception as e:
            logger.warning(
                f"Could not load stats for HTML report: {type(e).__name__}: {e}"
            )

        config_html = ""
        training_progress = self.get_training_process(output_dir)
        try:
            config_html = format_config_table_html(config, split_info, training_progress)
        except Exception as e:
            logger.warning(f"Could not load config for HTML report: {e}")

        def render_img_section(title: str, dir_path: Path, output_type: str = None) -> str:
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
                    [
                        img
                        for img in img_names.values()
                        if img.name not in display_order
                    ]
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

        button_html = """
        <button class="help-modal-btn openMetricsHelp">Model Evaluation Metrics — Help Guide</button>
        <br><br>
        <style>
        .help-modal-btn {
            background-color: #17623b;
            color: #fff;
            border: none;
            border-radius: 24px;
            padding: 10px 28px;
            font-size: 1.1rem;
            font-weight: bold;
            letter-spacing: 0.03em;
            cursor: pointer;
            transition: background 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 8px rgba(23,98,59,0.07);
        }
        .help-modal-btn:hover, .help-modal-btn:focus {
            background-color: #21895e;
            outline: none;
            box-shadow: 0 4px 16px rgba(23,98,59,0.14);
        }
        </style>
        """
        tab1_content = button_html + config_html + metrics_html
        tab2_content = (
            button_html
            + train_val_metrics_html
            + render_img_section("Training & Validation Visualizations", train_viz_dir)
        )
        tab3_content = (
            button_html
            + test_metrics_html
            + render_img_section("Test Visualizations", test_viz_dir, output_type)
        )

        tabbed_html = build_tabbed_html(tab1_content, tab2_content, tab3_content)
        modal_html = get_metrics_help_modal()
        html += tabbed_html + modal_html
        html += get_html_closing()

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

    def _prepare_data(self) -> Tuple[Path, Dict[str, Any]]:
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

        final_csv = TEMP_CSV_FILENAME
        try:
            df.to_csv(final_csv, index=False)
            logger.info(f"Saved prepared data to {final_csv}")
        except Exception:
            logger.error("Error saving prepared CSV", exc_info=True)
            raise

        return final_csv, split_config, split_info

    def _process_fixed_split(self, df: pd.DataFrame) -> Dict[str, Any]:
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
        help="Random split proportions (e.g., 0.7 0.1 0.2). Only used if no split column.",
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

    args = parser.parse_args()

    if not 0.0 <= args.validation_size <= 1.0:
        parser.error("validation-size must be between 0.0 and 1.0")
    if not args.csv_file.is_file():
        parser.error(f"CSV not found: {args.csv_file}")
    if not args.image_zip.is_file():
        parser.error(f"ZIP not found: {args.image_zip}")

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
