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
    <h2>Model Evaluation Metrics — Help Guide</h2>
    <div class="metrics-guide">
      <h3>1) General Metrics (Regression and Classification)</h3>
      <p><strong>Loss (Regression & Classification):</strong> Measures the difference between predicted and actual values, optimized during training. Lower is better. For regression, this is often Mean Squared Error (MSE) or Mean Absolute Error (MAE). For classification, it’s typically cross-entropy or log loss.</p>
      <h3>2) Regression Metrics</h3>
      <p><strong>Mean Absolute Error (MAE):</strong> Average of absolute differences between predicted and actual values, in the same units as the target. Use for interpretable error measurement when all errors are equally important. Less sensitive to outliers than MSE.</p>
      <p><strong>Mean Squared Error (MSE):</strong> Average of squared differences between predicted and actual values. Penalizes larger errors more heavily, useful when large deviations are critical. Often used as the loss function in regression.</p>
      <p><strong>Root Mean Squared Error (RMSE):</strong> Square root of MSE, in the same units as the target. Balances interpretability and sensitivity to large errors. Widely used for regression evaluation.</p>
      <p><strong>Mean Absolute Percentage Error (MAPE):</strong> Average absolute error as a percentage of actual values. Scale-independent, ideal for comparing relative errors across datasets. Avoid when actual values are near zero.</p>
      <p><strong>Root Mean Squared Percentage Error (RMSPE):</strong> Square root of mean squared percentage error. Scale-independent, penalizes larger relative errors more than MAPE. Use for forecasting or when relative accuracy matters.</p>
      <p><strong>R² Score:</strong> Proportion of variance in the target explained by the model. Ranges from negative infinity to 1 (perfect prediction). Use to assess model fit; negative values indicate poor performance compared to predicting the mean.</p>
      <h3>3) Classification Metrics</h3>
      <p><strong>Accuracy:</strong> Proportion of correct predictions among all predictions. Simple but misleading for imbalanced datasets, where high accuracy may hide poor performance on minority classes.</p>
      <p><strong>Micro Accuracy:</strong> Sums true positives and true negatives across all classes before computing accuracy. Suitable for multiclass or multilabel problems with imbalanced data.</p>
      <p><strong>Token Accuracy:</strong> Measures how often predicted tokens (e.g., in sequences) match true tokens. Common in NLP tasks like text generation or token classification.</p>
      <p><strong>Precision:</strong> Proportion of positive predictions that are correct (TP / (TP + FP)). Use when false positives are costly, e.g., spam detection.</p>
      <p><strong>Recall (Sensitivity):</strong> Proportion of actual positives correctly predicted (TP / (TP + FN)). Use when missing positives is risky, e.g., disease detection.</p>
      <p><strong>Specificity:</strong> True negative rate (TN / (TN + FP)). Measures ability to identify negatives. Useful in medical testing to avoid false alarms.</p>
      <h3>4) Classification: Macro, Micro, and Weighted Averages</h3>
      <p><strong>Macro Precision / Recall / F1:</strong> Averages the metric across all classes, treating each equally. Best for balanced datasets where all classes are equally important.</p>
      <p><strong>Micro Precision / Recall / F1:</strong> Aggregates true positives, false positives, and false negatives across all classes before computing. Ideal for imbalanced or multilabel classification.</p>
      <p><strong>Weighted Precision / Recall / F1:</strong> Averages metrics across classes, weighted by the number of true instances per class. Balances class importance based on frequency.</p>
      <h3>5) Classification: Average Precision (PR-AUC Variants)</h3>
      <p><strong>Average Precision Macro:</strong> Precision-Recall AUC averaged equally across classes. Use for balanced multiclass problems.</p>
      <p><strong>Average Precision Micro:</strong> Global Precision-Recall AUC using all instances. Best for imbalanced or multilabel classification.</p>
      <p><strong>Average Precision Samples:</strong> Precision-Recall AUC averaged across individual samples. Ideal for multilabel tasks where samples have multiple labels.</p>
      <h3>6) Classification: ROC-AUC Variants</h3>
      <p><strong>ROC-AUC:</strong> Measures ability to distinguish between classes. AUC = 1 is perfect; 0.5 is random guessing. Use for binary classification.</p>
      <p><strong>Macro ROC-AUC:</strong> Averages AUC across all classes equally. Suitable for balanced multiclass problems.</p>
      <p><strong>Micro ROC-AUC:</strong> Computes AUC from aggregated predictions across all classes. Useful for imbalanced or multilabel settings.</p>
      <h3>7) Classification: Confusion Matrix Stats (Per Class)</h3>
      <p><strong>True Positives / Negatives (TP / TN):</strong> Correct predictions for positives and negatives, respectively.</p>
      <p><strong>False Positives / Negatives (FP / FN):</strong> Incorrect predictions — false alarms and missed detections.</p>
      <h3>8) Classification: Ranking Metrics</h3>
      <p><strong>Hits at K:</strong> Measures whether the true label is among the top-K predictions. Common in recommendation systems and retrieval tasks.</p>
      <h3>9) Other Metrics (Classification)</h3>
      <p><strong>Cohen's Kappa:</strong> Measures agreement between predicted and actual labels, adjusted for chance. Useful for multiclass classification with imbalanced data.</p>
      <p><strong>Matthews Correlation Coefficient (MCC):</strong> Balanced measure using TP, TN, FP, and FN. Effective for imbalanced datasets.</p>
      <h3>10) Metric Recommendations</h3>
      <ul>
        <li><strong>Regression:</strong> Use <strong>RMSE</strong> or <strong>MAE</strong> for general evaluation, <strong>MAPE</strong> for relative errors, and <strong>R²</strong> to assess model fit. Use <strong>MSE</strong> or <strong>RMSPE</strong> when large errors are critical.</li>
        <li><strong>Classification (Balanced Data):</strong> Use <strong>Accuracy</strong> and <strong>F1</strong> for overall performance.</li>
        <li><strong>Classification (Imbalanced Data):</strong> Use <strong>Precision</strong>, <strong>Recall</strong>, and <strong>ROC-AUC</strong> to focus on minority class performance.</li>
        <li><strong>Multilabel or Imbalanced Classification:</strong> Use <strong>Micro Precision/Recall/F1</strong> or <strong>Micro ROC-AUC</strong>.</li>
        <li><strong>Balanced Multiclass:</strong> Use <strong>Macro Precision/Recall/F1</strong> or <strong>Macro ROC-AUC</strong>.</li>
        <li><strong>Class Frequency Matters:</strong> Use <strong>Weighted Precision/Recall/F1</strong> to account for class imbalance.</li>
        <li><strong>Recommendation/Ranking:</strong> Use <strong>Hits at K</strong> for retrieval tasks.</li>
        <li><strong>Detailed Analysis:</strong> Use <strong>Confusion Matrix stats</strong> for class-wise performance in classification.</li>
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
