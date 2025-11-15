import base64
import json
from typing import Any, Dict, List, Optional

from constants import METRIC_DISPLAY_NAMES
from utils import detect_output_type, extract_metrics_from_json

def generate_table_row(cells, styles):
    """Helper function to generate an HTML table row."""
    return (
        "<tr>"
        + "".join(f"<td style='{styles}'>{cell}</td>" for cell in cells)
        + "</tr>"
    )


def format_config_table_html(
    config: dict,
    split_info: Optional[str] = None,
    training_progress: dict = None,
    output_type: Optional[str] = None,
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
        "threshold",
    ]

    rows = []

    for key in display_keys:
        val = config.get(key, None)
        if key == "threshold":
            if output_type != "binary":
                continue
            val = val if val is not None else 0.5
            val_str = f"{val:.2f}"
            if val == 0.5:
                val_str += " (default)"
        else:
            if key == "task_type":
                val_str = val.title() if isinstance(val, str) else "N/A"
            elif key == "batch_size":
                if val is not None:
                    val_str = int(val)
                else:
                    val = "auto"
                    val_str = "auto"
            resolved_val = None
            if val is None or val == "auto":
                if training_progress:
                    resolved_val = training_progress.get("batch_size")
                    val = (
                        "Auto-selected batch size by Ludwig:<br>"
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
            elif key == "learning_rate":
                if val is not None and val != "auto":
                    val_str = f"{val:.6f}"
                else:
                    if training_progress:
                        resolved_val = training_progress.get("learning_rate")
                        val_str = (
                            "Auto-selected learning rate by Ludwig:<br>"
                            f"<span style='font-size: 0.85em;'>"
                            f"{resolved_val if resolved_val else 'auto'}</span><br>"
                            "<span style='font-size: 0.85em;'>"
                            "Based on model architecture and training setup "
                            "(e.g., fine-tuning).<br>"
                            "</span>"
                        )
                    else:
                        val_str = (
                            "Auto-selected by Ludwig<br>"
                            "<span style='font-size: 0.85em;'>"
                            "Automatically tuned based on architecture and dataset.<br>"
                            "See <a href='https://ludwig.ai/latest/configuration/trainer/"
                            "#trainer-parameters' target='_blank'>"
                            "Ludwig Trainer Parameters</a> for details."
                            "</span>"
                        )
            elif key == "epochs":
                if val is None:
                    val_str = "N/A"
                else:
                    if (
                        training_progress
                        and "epoch" in training_progress
                        and val > training_progress["epoch"]
                    ):
                        val_str = (
                            f"Because of early stopping: the training "
                            f"stopped at epoch {training_progress['epoch']}"
                        )
                    else:
                        val_str = val
            else:
                val_str = val if val is not None else "N/A"
            if val_str == "N/A" and key not in ["task_type"]:
                continue
        rows.append(
            f"<tr>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left; "
            f"white-space: normal; word-break: break-word; overflow-wrap: anywhere;'>"
            f"{key.replace('_', ' ').title()}</td>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center; "
            f"white-space: normal; word-break: break-word; overflow-wrap: anywhere;'>"
            f"{val_str}</td>"
            f"</tr>"
        )

    aug_cfg = config.get("augmentation")
    if aug_cfg:
        types = [str(a.get("type", "")) for a in aug_cfg]
        aug_val = ", ".join(types)
        rows.append(
            f"<tr><td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left; "
            f"white-space: normal; word-break: break-word; overflow-wrap: anywhere;'>Augmentation</td>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center; "
            f"white-space: normal; word-break: break-word; overflow-wrap: anywhere;'>{aug_val}</td></tr>"
        )

    if split_info:
        rows.append(
            f"<tr><td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left; "
            f"white-space: normal; word-break: break-word; overflow-wrap: anywhere;'>Data Split</td>"
            f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center; "
            f"white-space: normal; word-break: break-word; overflow-wrap: anywhere;'>{split_info}</td></tr>"
        )

    html = f"""
        <h2 style="text-align: center;">Model and Training Summary</h2>
        <div style="display: flex; justify-content: center;">
          <table style="border-collapse: collapse; width: 100%; table-layout: fixed;">
            <thead><tr>
              <th style="padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Parameter</th>
              <th style="padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Value</th>
            </tr></thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div><br>
        <p style="text-align: center; font-size: 0.9em;">
          Model trained using <a href="https://ludwig.ai/" target="_blank" rel="noopener noreferrer">Ludwig</a>.
          <a href="https://ludwig.ai/latest/configuration/" target="_blank" rel="noopener noreferrer">
            Ludwig documentation provides detailed information about default model and training parameters
          </a>
        </p><hr>
        """
    return html

def get_html_template():
    """
    Returns the opening HTML, <head> (with CSS/JS), and opens <body> + .container.
    Includes:
      - Base styling for layout and tables
      - Sortable table headers with 3-state arrows (none ⇅, asc ↑, desc ↓)
      - A scroll helper class (.scroll-rows-30) that approximates ~30 visible rows
      - A guarded script so initializing runs only once even if injected twice
    """
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Galaxy-Ludwig Report</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f4f4f4;
    }
    .container {
      max-width: 1200px;
      margin: auto;
      background: white;
      padding: 20px;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
      overflow-x: auto;
    }
    h1 {
      text-align: center;
      color: #333;
    }
    h2 {
      border-bottom: 2px solid #4CAF50;
      color: #4CAF50;
      padding-bottom: 5px;
      margin-top: 28px;
    }

    /* baseline table setup */
    table {
      border-collapse: collapse;
      margin: 20px 0;
      width: 100%;
      table-layout: fixed;
      background: #fff;
    }
    table, th, td {
      border: 1px solid #ddd;
    }
    th, td {
      padding: 10px;
      text-align: center;
      vertical-align: middle;
      word-break: break-word;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    th {
      background-color: #4CAF50;
      color: white;
    }

    .plot {
      text-align: center;
      margin: 20px 0;
    }
    .plot img {
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
    }

    /* -------------------
       sortable columns (3-state: none ⇅, asc ↑, desc ↓)
       ------------------- */
    table.performance-summary th.sortable {
      cursor: pointer;
      position: relative;
      user-select: none;
    }
    /* default icon space */
    table.performance-summary th.sortable::after {
      content: '⇅';
      position: absolute;
      right: 12px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 0.8em;
      color: #eaf5ea; /* light on green */
      text-shadow: 0 0 1px rgba(0,0,0,0.15);
    }
    /* three states override the default */
    table.performance-summary th.sortable.sorted-none::after { content: '⇅'; color: #eaf5ea; }
    table.performance-summary th.sortable.sorted-asc::after  { content: '↑';  color: #ffffff; }
    table.performance-summary th.sortable.sorted-desc::after { content: '↓';  color: #ffffff; }

    /* show ~30 rows with a scrollbar (tweak if you want) */
    .scroll-rows-30 {
      max-height: 900px;       /* ~30 rows depending on row height */
      overflow-y: auto;        /* vertical scrollbar ("sidebar") */
      overflow-x: auto;
    }

    /* Tabs + Help button (used by build_tabbed_html) */
    .tabs {
      display: flex;
      align-items: center;
      border-bottom: 2px solid #ccc;
      margin-bottom: 1rem;
      gap: 6px;
      flex-wrap: wrap;
    }
    .tab {
      padding: 10px 20px;
      cursor: pointer;
      border: 1px solid #ccc;
      border-bottom: none;
      background: #f9f9f9;
      margin-right: 5px;
      border-top-left-radius: 8px;
      border-top-right-radius: 8px;
    }
    .tab.active {
      background: white;
      font-weight: bold;
    }
    .help-btn {
      margin-left: auto;
      padding: 6px 12px;
      font-size: 0.9rem;
      border: 1px solid #4CAF50;
      border-radius: 4px;
      background: #4CAF50;
      color: white;
      cursor: pointer;
    }
    .tab-content {
      display: none;
      padding: 20px;
      border: 1px solid #ccc;
      border-top: none;
      background: #fff;
    }
    .tab-content.active {
      display: block;
    }

    /* Modal (used by get_metrics_help_modal) */
    .modal {
      display: none;
      position: fixed;
      z-index: 9999;
      left: 0; top: 0;
      width: 100%; height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.4);
    }
    .modal-content {
      background-color: #fefefe;
      margin: 8% auto;
      padding: 20px;
      border: 1px solid #888;
      width: 90%;
      max-width: 900px;
      border-radius: 8px;
    }
    .modal .close {
      color: #777;
      float: right;
      font-size: 28px;
      font-weight: bold;
      line-height: 1;
      margin-left: 8px;
    }
    .modal .close:hover,
    .modal .close:focus {
      color: black;
      text-decoration: none;
      cursor: pointer;
    }
    .metrics-guide h3 { margin-top: 20px; }
    .metrics-guide p { margin: 6px 0; }
    .metrics-guide ul { margin: 10px 0; padding-left: 20px; }
  </style>

  <script>
    // Guard to avoid double-initialization if this block is included twice
    (function(){
      if (window.__perfSummarySortInit) return;
      window.__perfSummarySortInit = true;

      function initPerfSummarySorting() {
        // Record original order for "back to original"
        document.querySelectorAll('table.performance-summary tbody').forEach(tbody => {
          Array.from(tbody.rows).forEach((row, i) => { row.dataset.originalOrder = i; });
        });

        const getText = td => (td?.innerText || '').trim();
        const cmp = (idx, asc) => (a, b) => {
          const v1 = getText(a.children[idx]);
          const v2 = getText(b.children[idx]);
          const n1 = parseFloat(v1), n2 = parseFloat(v2);
          if (!isNaN(n1) && !isNaN(n2)) return asc ? n1 - n2 : n2 - n1; // numeric
          return asc ? v1.localeCompare(v2) : v2.localeCompare(v1);       // lexical
        };

        document.querySelectorAll('table.performance-summary th.sortable').forEach(th => {
          // initialize to "none"
          th.classList.remove('sorted-asc','sorted-desc');
          th.classList.add('sorted-none');

          th.addEventListener('click', () => {
            const table = th.closest('table');
            const headerRow = th.parentNode;
            const allTh = headerRow.querySelectorAll('th.sortable');
            const tbody = table.querySelector('tbody');

            // Determine current state BEFORE clearing
            const isAsc  = th.classList.contains('sorted-asc');
            const isDesc = th.classList.contains('sorted-desc');

            // Reset all headers in this row
            allTh.forEach(x => x.classList.remove('sorted-asc','sorted-desc','sorted-none'));

            // Compute next state
            let next;
            if (!isAsc && !isDesc) {
              next = 'asc';
            } else if (isAsc) {
              next = 'desc';
            } else {
              next = 'none';
            }
            th.classList.add('sorted-' + next);

            // Sort rows according to the chosen state
            const rows = Array.from(tbody.rows);
            if (next === 'none') {
              rows.sort((a, b) => (a.dataset.originalOrder - b.dataset.originalOrder));
            } else {
              const idx = Array.from(headerRow.children).indexOf(th);
              rows.sort(cmp(idx, next === 'asc'));
            }
            rows.forEach(r => tbody.appendChild(r));
          });
        });
      }

      // Run after DOM is ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPerfSummarySorting);
      } else {
        initPerfSummarySorting();
      }
    })();
  </script>
</head>
<body>
  <div class="container">
"""


def get_html_closing():
    """Closes .container, body, and html."""
    return """
  </div>
</body>
</html>
"""


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def json_to_nested_html_table(json_data, depth: int = 0) -> str:
    """
    Convert a JSON-able object to an HTML nested table.
    Renders dicts as two-column tables (key/value) and lists as index/value rows.
    """
    # Base case: flat dict (no nested dict/list values)
    if isinstance(json_data, dict) and all(
        not isinstance(v, (dict, list)) for v in json_data.values()
    ):
        rows = [
            f"<tr><th>{key}</th><td>{value}</td></tr>"
            for key, value in json_data.items()
        ]
        return f"<table>{''.join(rows)}</table>"

    # Base case: list of simple values
    if isinstance(json_data, list) and all(
        not isinstance(v, (dict, list)) for v in json_data
    ):
        rows = [
            f"<tr><th>Index {i}</th><td>{value}</td></tr>"
            for i, value in enumerate(json_data)
        ]
        return f"<table>{''.join(rows)}</table>"

    # Recursive cases
    if isinstance(json_data, dict):
        rows = [
            (
                f"<tr><th style='text-align:left;padding-left:{depth * 20}px;'>{key}</th>"
                f"<td>{json_to_nested_html_table(value, depth + 1)}</td></tr>"
            )
            for key, value in json_data.items()
        ]
        return f"<table>{''.join(rows)}</table>"

    if isinstance(json_data, list):
        rows = [
            (
                f"<tr><th style='text-align:left;padding-left:{depth * 20}px;'>[{i}]</th>"
                f"<td>{json_to_nested_html_table(value, depth + 1)}</td></tr>"
            )
            for i, value in enumerate(json_data)
        ]
        return f"<table>{''.join(rows)}</table>"

    # Primitive
    return f"{json_data}"


def json_to_html_table(json_data) -> str:
    """
    Convert JSON (dict or string) into a vertically oriented HTML table.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    return json_to_nested_html_table(json_data)


def build_tabbed_html(metrics_html: str, train_val_html: str, test_html: str) -> str:
    """
    Build a 3-tab interface:
      - Config and Results Summary
      - Train/Validation Results
      - Test Results
    Includes a persistent "Help" button that toggles the metrics modal.
    """
    return f"""
<div class="tabs">
  <div class="tab active" onclick="showTab('metrics')">Config and Results Summary</div>
  <div class="tab" onclick="showTab('trainval')">Train/Validation Results</div>
  <div class="tab" onclick="showTab('test')">Test Results</div>
  <button id="openMetricsHelp" class="help-btn" title="Open metrics help">Help</button>
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
    // find tab with matching onclick target
    document.querySelectorAll('.tab').forEach(t => {{
      if (t.getAttribute('onclick') && t.getAttribute('onclick').includes(id)) {{
        t.classList.add('active');
      }}
    }});
  }}
</script>
"""


def get_metrics_help_modal() -> str:
    """
    Returns a ready-to-use modal with a comprehensive metrics guide and
    the small script that wires the "Help" button to open/close the modal.
    """
    modal_html = (
        '<div id="metricsHelpModal" class="modal">'
        '  <div class="modal-content">'
        '    <span class="close">×</span>'
        "    <h2>Model Evaluation Metrics — Help Guide</h2>"
        '    <div class="metrics-guide">'
        '      <h3>1) General Metrics (Regression and Classification)</h3>'
        '      <p><strong>Loss (Regression & Classification):</strong> '
        'Measures the difference between predicted and actual values, '
        'optimized during training. Lower is better. '
        'For regression, this is often Mean Squared Error (MSE) or '
        'Mean Absolute Error (MAE). For classification, it\'s typically '
        'cross-entropy or log loss.</p>'
        '      <h3>2) Regression Metrics</h3>'
        '      <p><strong>Mean Absolute Error (MAE):</strong> '
        'Average of absolute differences between predicted and actual values, '
        'in the same units as the target. Use for interpretable error measurement '
        'when all errors are equally important. Less sensitive to outliers than MSE.</p>'
        '      <p><strong>Mean Squared Error (MSE):</strong> '
        'Average of squared differences between predicted and actual values. '
        'Penalizes larger errors more heavily, useful when large deviations are critical. '
        'Often used as the loss function in regression.</p>'
        '      <p><strong>Root Mean Squared Error (RMSE):</strong> '
        'Square root of MSE, in the same units as the target. '
        'Balances interpretability and sensitivity to large errors. '
        'Widely used for regression evaluation.</p>'
        '      <p><strong>Mean Absolute Percentage Error (MAPE):</strong> '
        'Average absolute error as a percentage of actual values. '
        'Scale-independent, ideal for comparing relative errors across datasets. '
        'Avoid when actual values are near zero.</p>'
        '      <p><strong>Root Mean Squared Percentage Error (RMSPE):</strong> '
        'Square root of mean squared percentage error. Scale-independent, '
        'penalizes larger relative errors more than MAPE. Use for forecasting '
        'or when relative accuracy matters.</p>'
        '      <p><strong>R² Score:</strong> Proportion of variance in the target '
        'explained by the model. Ranges from negative infinity to 1 (perfect prediction). '
        'Use to assess model fit; negative values indicate poor performance '
        'compared to predicting the mean.</p>'
        '      <h3>3) Classification Metrics</h3>'
        '      <p><strong>Accuracy:</strong> Proportion of correct predictions '
        'among all predictions. Simple but misleading for imbalanced datasets, '
        'where high accuracy may hide poor performance on minority classes.</p>'
        '      <p><strong>Micro Accuracy:</strong> Sums true positives and true negatives '
        'across all classes before computing accuracy. Suitable for multiclass or '
        'multilabel problems with imbalanced data.</p>'
        '      <p><strong>Token Accuracy:</strong> Measures how often predicted tokens '
        '(e.g., in sequences) match true tokens. Common in NLP tasks like text generation '
        'or token classification.</p>'
        '      <p><strong>Precision:</strong> Proportion of positive predictions that are '
        'correct (TP / (TP + FP)). Use when false positives are costly, e.g., spam detection.</p>'
        '      <p><strong>Recall (Sensitivity):</strong> Proportion of actual positives '
        'correctly predicted (TP / (TP + FN)). Use when missing positives is risky, '
        'e.g., disease detection.</p>'
        '      <p><strong>Specificity:</strong> True negative rate (TN / (TN + FP)). '
        'Measures ability to identify negatives. Useful in medical testing to avoid '
        'false alarms.</p>'
        '      <h3>4) Classification: Macro, Micro, and Weighted Averages</h3>'
        '      <p><strong>Macro Precision / Recall / F1:</strong> Averages the metric '
        'across all classes, treating each equally. Best for balanced datasets where '
        'all classes are equally important.</p>'
        '      <p><strong>Micro Precision / Recall / F1:</strong> Aggregates true positives, '
        'false positives, and false negatives across all classes before computing. '
        'Ideal for imbalanced or multilabel classification.</p>'
        '      <p><strong>Weighted Precision / Recall / F1:</strong> Averages metrics '
        'across classes, weighted by the number of true instances per class. Balances '
        'class importance based on frequency.</p>'
        '      <h3>5) Classification: Average Precision (PR-AUC Variants)</h3>'
        '      <p><strong>Average Precision Macro:</strong> Precision-Recall AUC averaged '
        'equally across classes. Use for balanced multiclass problems.</p>'
        '      <p><strong>Average Precision Micro:</strong> Global Precision-Recall AUC '
        'using all instances. Best for imbalanced or multilabel classification.</p>'
        '      <p><strong>Average Precision Samples:</strong> Precision-Recall AUC averaged '
        'across individual samples. Ideal for multilabel tasks where samples have multiple '
        'labels.</p>'
        '      <h3>6) Classification: ROC-AUC Variants</h3>'
        '      <p><strong>ROC-AUC:</strong> Measures ability to distinguish between classes. '
        'AUC = 1 is perfect; 0.5 is random guessing. Use for binary classification.</p>'
        '      <p><strong>Macro ROC-AUC:</strong> Averages AUC across all classes equally. '
        'Suitable for balanced multiclass problems.</p>'
        '      <p><strong>Micro ROC-AUC:</strong> Computes AUC from aggregated predictions '
        'across all classes. Useful for imbalanced or multilabel settings.</p>'
        '      <h3>7) Classification: Confusion Matrix Stats (Per Class)</h3>'
        '      <p><strong>True Positives / Negatives (TP / TN):</strong> Correct predictions '
        'for positives and negatives, respectively.</p>'
        '      <p><strong>False Positives / Negatives (FP / FN):</strong> Incorrect predictions '
        '— false alarms and missed detections.</p>'
        '      <h3>8) Classification: Ranking Metrics</h3>'
        '      <p><strong>Hits at K:</strong> Measures whether the true label is among the '
        'top-K predictions. Common in recommendation systems and retrieval tasks.</p>'
        '      <h3>9) Other Metrics (Classification)</h3>'
        '      <p><strong>Cohen\'s Kappa:</strong> Measures agreement between predicted and '
        'actual labels, adjusted for chance. Useful for multiclass classification with '
        'imbalanced data.</p>'
        '      <p><strong>Matthews Correlation Coefficient (MCC):</strong> Balanced measure '
        'using TP, TN, FP, and FN. Effective for imbalanced datasets.</p>'
        '      <h3>10) Metric Recommendations</h3>'
        '      <ul>'
        '        <li><strong>Regression:</strong> Use <strong>RMSE</strong> or '
        '<strong>MAE</strong> for general evaluation, <strong>MAPE</strong> for relative '
        'errors, and <strong>R²</strong> to assess model fit. Use <strong>MSE</strong> or '
        '<strong>RMSPE</strong> when large errors are critical.</li>'
        '        <li><strong>Classification (Balanced Data):</strong> Use <strong>Accuracy</strong> '
        'and <strong>F1</strong> for overall performance.</li>'
        '        <li><strong>Classification (Imbalanced Data):</strong> Use <strong>Precision</strong>, '
        '<strong>Recall</strong>, and <strong>ROC-AUC</strong> to focus on minority class '
        'performance.</li>'
        '        <li><strong>Multilabel or Imbalanced Classification:</strong> Use '
        '<strong>Micro Precision/Recall/F1</strong> or <strong>Micro ROC-AUC</strong>.</li>'
        '        <li><strong>Balanced Multiclass:</strong> Use <strong>Macro Precision/Recall/F1</strong> '
        'or <strong>Macro ROC-AUC</strong>.</li>'
        '        <li><strong>Class Frequency Matters:</strong> Use <strong>Weighted Precision/Recall/F1</strong> '
        'to account for class imbalance.</li>'
        '        <li><strong>Recommendation/Ranking:</strong> Use <strong>Hits at K</strong> for retrieval tasks.</li>'
        '        <li><strong>Detailed Analysis:</strong> Use <strong>Confusion Matrix stats</strong> '
        'for class-wise performance in classification.</li>'
        '      </ul>'
        '    </div>'
        '  </div>'
        '</div>'
    )

    modal_js = (
        "<script>"
        "document.addEventListener('DOMContentLoaded', function() {"
        "  var modal = document.getElementById('metricsHelpModal');"
        "  var openBtn = document.getElementById('openMetricsHelp');"
        "  var closeBtn = modal ? modal.querySelector('.close') : null;"
        "  if (openBtn && modal) {"
        "    openBtn.addEventListener('click', function(){ modal.style.display = 'block'; });"
        "  }"
        "  if (closeBtn && modal) {"
        "    closeBtn.addEventListener('click', function(){ modal.style.display = 'none'; });"
        "  }"
        "  window.addEventListener('click', function(ev){"
        "    if (ev.target === modal) { modal.style.display = 'none'; }"
        "  });"
        "});"
        "</script>"
    )
    return modal_html + modal_js

# -----------------------------------------
# MODEL PERFORMANCE (Train/Val/Test) TABLE
# -----------------------------------------
def format_stats_table_html(train_stats: dict, test_stats: dict, output_type: str) -> str:
    """Formats a combined HTML table for training, validation, and test metrics."""
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
        "<table class='performance-summary' style='border-collapse: collapse;'>"
        "<thead><tr>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Train</th>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Validation</th>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Test</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;",
        )
    html += "</tbody></table></div><br>"
    return html

# -------------------------------------------
# TRAIN/VALIDATION PERFORMANCE SUMMARY TABLE
# -------------------------------------------
def format_train_val_stats_table_html(train_stats: dict, test_stats: dict) -> str:
    """Format train/validation metrics into an HTML table."""
    all_metrics = extract_metrics_from_json(train_stats, test_stats, detect_output_type(test_stats))
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
        "<table class='performance-summary' style='border-collapse: collapse;'>"
        "<thead><tr>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Train</th>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Validation</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;",
        )
    html += "</tbody></table></div><br>"
    return html

# -----------------------------------------
# TEST‐ONLY PERFORMANCE SUMMARY TABLE
# -----------------------------------------
def format_test_merged_stats_table_html(
    test_metrics: Dict[str, Any], output_type: str
) -> str:
    """Format test metrics into an HTML table."""
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
        "<table class='performance-summary' style='border-collapse: collapse;'>"
        "<thead><tr>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>"
        "<th class='sortable' style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Test</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;",
        )
    html += "</tbody></table></div><br>"
    return html
