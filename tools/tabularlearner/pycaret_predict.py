import argparse
import logging
import tempfile

import h5py
import joblib
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from sklearn.metrics import average_precision_score

from utils import encode_image_to_base64, get_html_closing, get_html_template

LOG = logging.getLogger(__name__)


class PyCaretModelEvaluator:
    def __init__(self, model_path, task, target):
        self.model_path = model_path
        self.task = task.lower()
        self.model = self.load_h5_model()
        self.target = target if target != "None" else None

    def load_h5_model(self):
        """Load a PyCaret model from an HDF5 file."""
        with h5py.File(self.model_path, 'r') as f:
            model_bytes = bytes(f['model'][()])
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(model_bytes)
                temp_file.seek(0)
                loaded_model = joblib.load(temp_file.name)
        return loaded_model

    def evaluate(self, data_path):
        """Evaluate the model using the specified data."""
        raise NotImplementedError("Subclasses must implement this method")


class ClassificationEvaluator(PyCaretModelEvaluator):
    def evaluate(self, data_path):
        metrics = None
        plot_paths = {}
        data = pd.read_csv(data_path, engine='python', sep=None)
        if self.target:
            exp = ClassificationExperiment()
            names = data.columns.to_list()
            LOG.error(f"Column names: {names}")
            target_index = int(self.target) - 1
            target_name = names[target_index]
            exp.setup(data, target=target_name, test_data=data, index=False)
            exp.add_metric(id='PR-AUC-Weighted',
                           name='PR-AUC-Weighted',
                           target='pred_proba',
                           score_func=average_precision_score,
                           average='weighted')
            predictions = exp.predict_model(self.model)
            metrics = exp.pull()
            plots = ['confusion_matrix', 'auc', 'threshold', 'pr',
                     'error', 'class_report', 'learning', 'calibration',
                     'vc', 'dimension', 'manifold', 'rfe', 'feature',
                     'feature_all']
            for plot_name in plots:
                try:
                    if plot_name == 'auc' and not exp.is_multiclass:
                        plot_path = exp.plot_model(self.model,
                                                   plot=plot_name,
                                                   save=True,
                                                   plot_kwargs={
                                                       'micro': False,
                                                       'macro': False,
                                                       'per_class': False,
                                                       'binary': True})
                        plot_paths[plot_name] = plot_path
                        continue

                    plot_path = exp.plot_model(self.model,
                                               plot=plot_name, save=True)
                    plot_paths[plot_name] = plot_path
                except Exception as e:
                    LOG.error(f"Error generating plot {plot_name}: {e}")
                    continue
            generate_html_report(plot_paths, metrics)

        else:
            exp = ClassificationExperiment()
            exp.setup(data, target=None, test_data=data, index=False)
            predictions = exp.predict_model(self.model, data=data)

        return predictions, metrics, plot_paths


class RegressionEvaluator(PyCaretModelEvaluator):
    def evaluate(self, data_path):
        metrics = None
        plot_paths = {}
        data = pd.read_csv(data_path, engine='python', sep=None)
        if self.target:
            names = data.columns.to_list()
            target_index = int(self.target) - 1
            target_name = names[target_index]
            exp = RegressionExperiment()
            exp.setup(data, target=target_name, test_data=data, index=False)
            predictions = exp.predict_model(self.model)
            metrics = exp.pull()
            plots = ['residuals', 'error', 'cooks',
                     'learning', 'vc', 'manifold',
                     'rfe', 'feature', 'feature_all']
            for plot_name in plots:
                try:
                    plot_path = exp.plot_model(self.model,
                                               plot=plot_name, save=True)
                    plot_paths[plot_name] = plot_path
                except Exception as e:
                    LOG.error(f"Error generating plot {plot_name}: {e}")
                    continue
            generate_html_report(plot_paths, metrics)
        else:
            exp = RegressionExperiment()
            exp.setup(data, target=None, test_data=data, index=False)
            predictions = exp.predict_model(self.model, data=data)

        return predictions, metrics, plot_paths


def generate_html_report(plots, metrics):
    """Generate an HTML evaluation report."""
    plots_html = ""
    for plot_name, plot_path in plots.items():
        encoded_image = encode_image_to_base64(plot_path)
        plots_html += f"""
        <div class="plot">
            <h3>{plot_name.capitalize()}</h3>
            <img src="data:image/png;base64,{encoded_image}" alt="{plot_name}">
        </div>
        <hr>
        """

    metrics_html = metrics.to_html(index=False, classes="table")

    html_content = f"""
    {get_html_template()}
    <h1>Model Evaluation Report</h1>
    <div class="tabs">
        <div class="tab" onclick="openTab(event, 'metrics')">Metrics</div>
        <div class="tab" onclick="openTab(event, 'plots')">Plots</div>
    </div>
    <div id="metrics" class="tab-content">
        <h2>Metrics</h2>
        <table>
            {metrics_html}
        </table>
    </div>
    <div id="plots" class="tab-content">
        <h2>Plots</h2>
        {plots_html}
    </div>
    {get_html_closing()}
    """

    # Save HTML report
    with open("evaluation_report.html", "w") as html_file:
        html_file.write(html_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PyCaret model stored in HDF5 format.")
    parser.add_argument("--model_path",
                        type=str,
                        help="Path to the HDF5 model file.")
    parser.add_argument("--data_path",
                        type=str,
                        help="Path to the evaluation data CSV file.")
    parser.add_argument("--task",
                        type=str,
                        choices=["classification", "regression"],
                        help="Specify the task: classification or regression.")
    parser.add_argument("--target",
                        default=None,
                        help="Column number of the target")
    args = parser.parse_args()

    if args.task == "classification":
        evaluator = ClassificationEvaluator(
            args.model_path, args.task, args.target)
    elif args.task == "regression":
        evaluator = RegressionEvaluator(
            args.model_path, args.task, args.target)
    else:
        raise ValueError(
            "Unsupported task type. Use 'classification' or 'regression'.")

    predictions, metrics, plots = evaluator.evaluate(args.data_path)

    predictions.to_csv("predictions.csv", index=False)
