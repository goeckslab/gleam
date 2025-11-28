import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
def get_metrics_help_modal() -> str:
    # The HTML structure of the modal
    modal_html = """
<div id="metricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close">×</span>
    <h2>How to read this Multimodal Learner report</h2>
    <div class="metrics-guide">
      <h3>Tabs & layout</h3>
      <p><strong>Model Metric Summary and Config:</strong> Top-level metrics and the key run settings (target column, backbones, presets).</p>
      <p><strong>Train and Validation Summary:</strong> Learning curves plus combined ROC/PR/Calibration (binary), and any remaining diagnostics.</p>
      <p><strong>Test Summary:</strong> Test metrics table followed by the ROC/PR charts with your chosen threshold marked, and the Prediction Confidence histogram.</p>

      <h3>Dataset Overview</h3>
      <p>Shows label counts across Train/Validation/Test so you can quickly spot imbalance or missing splits.</p>

      <h3>Learning curves</h3>
      <p><strong>Label Accuracy & Loss:</strong> Train (blue) and Validation (orange) trends. Parallel curves that plateau suggest stable training; large gaps can indicate overfitting.</p>

      <h3>Binary diagnostics (Train vs Validation)</h3>
      <p><strong>ROC Curve:</strong> Both splits on one plot. Higher and leftward is better. The red “x” marks the decision threshold when provided.</p>
      <p><strong>Precision–Recall:</strong> Both splits on one plot; more informative on imbalance. Red marker shows the threshold point.</p>
      <p><strong>Calibration:</strong> Ideally near the diagonal; deviations show over/under-confidence.</p>
      <p><strong>Threshold Plot (Validation):</strong> Explore precision/recall/F1 vs threshold; use to pick a balanced operating point.</p>

      <h3>Test tab highlights</h3>
      <p><strong>Metrics table:</strong> Thresholded metrics for the test set.</p>
      <p><strong>ROC & PR:</strong> Thick lines, red marker and annotation for the selected threshold.</p>
      <p><strong>Prediction Confidence:</strong> Histogram of max predicted probabilities (as % of samples) to spot over/under-confidence.</p>

      <h3>Threshold tips</h3>
      <ul>
        <li>Use the Validation curves to choose a threshold that balances precision/recall for your use case.</li>
        <li>Threshold marker/annotation appears on ROC/PR plots when you pass <code>--threshold</code> (binary tasks).</li>
      </ul>

      <h3>When to worry</h3>
      <ul>
        <li>Huge train/val gaps on learning curves → possible overfitting.</li>
        <li>Calibration far from diagonal → predicted probabilities may be poorly calibrated.</li>
        <li>Very imbalanced label counts → focus on PR curves and per-class metrics (if enabled).</li>
      </ul>
    </div>
  </div>
</div>
"""
    # The CSS needed to style and hide/show the modal
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
    # The JavaScript to open/close the modal on button click
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

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def generate_feature_importance(*args, **kwargs):
    return "<p><em>Feature importance visualizations are not supported for this MultiModal workflow.</em></p>"
