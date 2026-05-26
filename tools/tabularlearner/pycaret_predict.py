import argparse
import logging
import tempfile

import h5py
import joblib
import numpy as np
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from sklearn.metrics import auc, precision_recall_curve
from utils import (
    build_tabbed_html,
    encode_image_to_base64,
    get_html_closing,
    get_html_template,
)


LOG = logging.getLogger(__name__)


MULTICLASS_UNAVAILABLE_PYCARET_PLOTS = {"threshold"}


def _should_skip_pycaret_plot(plot_name, exp):
    return (
        getattr(exp, "is_multiclass", False)
        and plot_name in MULTICLASS_UNAVAILABLE_PYCARET_PLOTS
    )


def _weighted_ovr_pr_auc(y_true, y_score, labels=None):
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    if labels is not None:
        class_labels = list(labels)
    else:
        class_labels = list(pd.unique(y_true_series))
        try:
            class_labels = sorted(class_labels)
        except Exception:
            pass
    if len(class_labels) < 2:
        return np.nan

    scores = np.asarray(y_score)
    if len(scores) != len(y_true_series):
        return np.nan

    if len(class_labels) == 2:
        try:
            pos_label = 1 if 1 in class_labels else sorted(class_labels)[-1]
        except Exception:
            pos_label = class_labels[-1]
        if scores.ndim == 2:
            if scores.shape[1] < 2:
                scores = scores.ravel()
            else:
                try:
                    pos_idx = class_labels.index(pos_label)
                except ValueError:
                    pos_idx = scores.shape[1] - 1
                scores = scores[:, min(pos_idx, scores.shape[1] - 1)]
        precision, recall, _ = precision_recall_curve(
            (y_true_series == pos_label).astype(int),
            scores,
        )
        return auc(recall, precision)

    if scores.ndim != 2 or scores.shape[1] < len(class_labels):
        return np.nan

    weighted_total = 0.0
    support_total = 0
    for class_idx, class_label in enumerate(class_labels):
        y_true_bin = (y_true_series == class_label).astype(int)
        if len(pd.unique(y_true_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(
            y_true_bin,
            scores[:, class_idx],
        )
        support = int(y_true_bin.sum())
        weighted_total += auc(recall, precision) * support
        support_total += support

    return weighted_total / support_total if support_total else np.nan


def pr_auc_curve_score(y_true, y_score):
    return _weighted_ovr_pr_auc(y_true, y_score)


def _add_pr_auc_metric_if_supported(exp):
    if getattr(exp, "is_multiclass", False):
        LOG.info(
            "Skipping PyCaret custom PR-AUC metric for multiclass "
            "classification."
        )
        return False
    exp.add_metric(id='PR-AUC',
                   name='PR-AUC',
                   target='pred_proba',
                   score_func=pr_auc_curve_score)
    return True


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
            LOG.info(f"Column names: {names}")
            target_index = int(self.target) - 1
            target_name = names[target_index]
            exp.setup(data, target=target_name, test_data=data, index=False)
            _add_pr_auc_metric_if_supported(exp)
            predictions = exp.predict_model(self.model)
            metrics = exp.pull()
            plots = ['confusion_matrix', 'auc', 'threshold', 'pr',
                     'error', 'class_report', 'learning', 'calibration',
                     'vc', 'dimension', 'manifold', 'rfe', 'feature',
                     'feature_all']
            for plot_name in plots:
                try:
                    if _should_skip_pycaret_plot(plot_name, exp):
                        LOG.info(
                            "Skipping PyCaret %s plot for multiclass "
                            "classification.",
                            plot_name,
                        )
                        continue
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

    html_content = (
        get_html_template()
        + "<h1>Model Evaluation Report</h1>"
        + build_tabbed_html(
            "<h2>Metrics</h2><div class='table-wrapper'>" + metrics_html + "</div>",
            "<h2>Plots</h2>" + plots_html,
            None,
            summary_tab_label="Metrics",
            test_tab_label="Plots",
        )
        + get_html_closing()
    )

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
