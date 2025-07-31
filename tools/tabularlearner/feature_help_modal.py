def get_feature_metrics_help_modal() -> str:
    modal_html = """
<div id="metricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close">&times;</span>
    <h2>Help Guide: Common Model Metrics</h2>
    <div class="metrics-guide">

      <!-- Classification Metrics -->
      <h3>1) Classification Metrics</h3>

      <p><strong>Accuracy:</strong>
      The proportion of correct predictions over all predictions:<br>
      <code>(TP + TN) / (TP + TN + FP + FN)</code>.
      <em>Use when</em> classes are balanced and you want a single easy‐to‐interpret number.</p>

      <p><strong>Precision:</strong>
      The fraction of positive predictions that are actually positive:<br>
      <code>TP / (TP + FP)</code>.
      <em>Use when</em> false positives are costly (e.g. spam filter—better to miss some spam than flag good mail).</p>

      <p><strong>Recall (Sensitivity):</strong>
      The fraction of actual positives correctly identified:<br>
      <code>TP / (TP + FN)</code>
      <em>Use when</em> false negatives are costly (e.g. disease screening—don’t miss sick patients).</p>

      <p><strong>F1 Score:</strong>
      The harmonic mean of Precision and Recall:<br>
      <code>2·(Precision·Recall)/(Precision+Recall)</code>
      <em>Use when</em> you need a balance between Precision & Recall on an imbalanced dataset.</p>

      <p><strong>ROC-AUC (Area Under ROC Curve):</strong>
      Measures ability to distinguish classes across all thresholds.
      Ranges from 0.5 (random) to 1 (perfect).
      <em>Use when</em> you care about ranking positives above negatives.</p>

      <p><strong>PR-AUC (Area Under Precision-Recall Curve):</strong>
      Summarizes Precision vs. Recall trade-off.
      More informative than ROC-AUC when positives are rare.
      <em>Use when</em> dealing with highly imbalanced data.</p>

      <p><strong>Log Loss:</strong>
      Penalizes confident wrong predictions via negative log-likelihood.
      Lower is better.
      <em>Use when</em> you need well-calibrated probability estimates.</p>

      <p><strong>Cohen’s Kappa:</strong>
      Measures agreement between predictions and true labels accounting for chance.
      1 is perfect, 0 is random.
      <em>Use when</em> you want to factor out chance agreement.</p>

      <hr>

      <!-- Regression Metrics -->
      <h3>2) Regression Metrics</h3>

      <p><strong>R² (Coefficient of Determination):</strong>
      Proportion of variance in the target explained by features:<br>
      1 is perfect, 0 means no better than predicting the mean, negative is worse than mean.
      <em>Use when</em> you want a normalized measure of fit.</p>

      <p><strong>MAE (Mean Absolute Error):</strong>
      Average absolute difference between predictions and actual values:<br>
      <code>mean(|y_pred − y_true|)</code>
      <em>Use when</em> you need an interpretable “average” error and want to downweight outliers.</p>

      <p><strong>RMSE (Root Mean Squared Error):</strong>
      Square root of the average squared errors:<br>
      <code>√mean((y_pred − y_true)²)</code>.
      Penalizes large errors more heavily.
      <em>Use when</em> large deviations are especially undesirable.</p>

      <p><strong>MSE (Mean Squared Error):</strong>
      The average squared error:<br>
      <code>mean((y_pred − y_true)²)</code>.
      Similar to RMSE but in squared units; often used in optimization.</p>

      <p><strong>RMSLE (Root Mean Squared Log Error):</strong>
      <code>√mean((log(1+y_pred) − log(1+y_true))²)</code>.
      Less sensitive to large differences when both true and predicted are large.
      <em>Use when</em> target spans several orders of magnitude.</p>

      <p><strong>MAPE (Mean Absolute Percentage Error):</strong>
      <code>mean(|(y_true − y_pred)/y_true|)·100</code>.
      Expresses error as a percentage.
      <em>Use when</em> relative error matters—but avoid if y_true≈0.</p>

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
