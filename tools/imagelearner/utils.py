import base64
import json


def get_html_template():
    return """
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
              max-width: 800px;
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
          }
          table {
              border-collapse: collapse;
              margin: 20px 0;
              width: 100%;
              table-layout: fixed; /* Enforces consistent column widths */
          }
          table, th, td {
              border: 1px solid #ddd;
          }
          th, td {
              padding: 8px;
              text-align: center; /* Center-align text */
              vertical-align: middle; /* Center-align content vertically */
              word-wrap: break-word; /* Break long words to avoid overflow */
          }
          th:first-child, td:first-child {
              width: 5%; /* Smaller width for the first column */
          }
          th:nth-child(2), td:nth-child(2) {
              width: 50%; /* Wider for the metric/description column */
          }
          th:last-child, td:last-child {
              width: 25%; /* Value column gets remaining space */
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
          }
        </style>
    </head>
    <body>
    <div class="container">
    """


def get_html_closing():
    return """
    </div>
    </body>
    </html>
    """


def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def json_to_nested_html_table(json_data, depth=0):
    """
    Convert JSON object to an HTML nested table.

    Parameters:
        json_data (dict or list): The JSON data to convert.
        depth (int): Current depth level for indentation.

    Returns:
        str: HTML string for the nested table.
    """
    # Base case: if JSON is a simple key-value pair dictionary
    if isinstance(json_data, dict) and all(
        not isinstance(v, (dict, list)) for v in json_data.values()
    ):
        # Render a flat table
        rows = [
            f"<tr><th>{key}</th><td>{value}</td></tr>"
            for key, value in json_data.items()
        ]
        return f"<table>{''.join(rows)}</table>"

    # Base case: if JSON is a list of simple values
    if isinstance(json_data, list) and all(
        not isinstance(v, (dict, list)) for v in json_data
    ):
        rows = [
            f"<tr><th>Index {i}</th><td>{value}</td></tr>"
            for i, value in enumerate(json_data)
        ]
        return f"<table>{''.join(rows)}</table>"

    # Recursive case: if JSON contains nested structures
    if isinstance(json_data, dict):
        rows = [
            f"<tr><th style='padding-left:{depth * 20}px;'>{key}</th>"
            f"<td>{json_to_nested_html_table(value, depth + 1)}</td></tr>"
            for key, value in json_data.items()
        ]
        return f"<table>{''.join(rows)}</table>"

    if isinstance(json_data, list):
        rows = [
            f"<tr><th style='padding-left:{depth * 20}px;'>[{i}]</th>"
            f"<td>{json_to_nested_html_table(value, depth + 1)}</td></tr>"
            for i, value in enumerate(json_data)
        ]
        return f"<table>{''.join(rows)}</table>"

    # Base case: simple value
    return f"{json_data}"


def json_to_html_table(json_data):
    """
    Convert JSON to a vertically oriented HTML table.

    Parameters:
        json_data (str or dict): JSON string or dictionary.

    Returns:
        str: HTML table representation.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    return json_to_nested_html_table(json_data)


def build_tabbed_html(metrics_html: str, train_val_html: str, test_html: str) -> str:
    return f"""
<style>
  .tabs {{
    display: flex;
    align-items: center;
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
  /* new help-button styling */
  .help-btn {{
    margin-left: auto;
    padding: 6px 12px;
    font-size: 0.9rem;
    border: 1px solid #4CAF50;
    border-radius: 4px;
    background: #4CAF50;
    color: white;
    cursor: pointer;
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
  <div class="tab active" onclick="showTab('metrics')">Config &amp; Results Summary</div>
  <div class="tab" onclick="showTab('trainval')">Train/Validation Results</div>
  <div class="tab" onclick="showTab('test')">Test Results</div>
  <!-- always-visible help button -->
  <button id="openMetricsHelp" class="help-btn">Help</button>
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
  var openBtn = document.getElementById("openMetricsHelp");
  var span = document.getElementsByClassName("close")[0];
  if (openBtn && modal) {
    openBtn.onclick = function() {
      modal.style.display = "block";
    };
  }
  if (span && modal) {
    span.onclick = function() {
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
