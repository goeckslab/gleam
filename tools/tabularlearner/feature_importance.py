import base64
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(
            self,
            task_type,
            output_dir,
            data_path=None,
            data=None,
            target_col=None,
            exp=None,
            best_model=None):

        self.task_type = task_type
        self.output_dir = output_dir
        self.exp = exp
        self.best_model = best_model

        if exp is not None:
            # Assume all configs (data, target) are in exp
            self.data = exp.dataset.copy()
            self.target = exp.target_param
            LOG.info("Using provided experiment object")
        else:
            if data is not None:
                self.data = data
                LOG.info("Data loaded from memory")
            else:
                self.target_col = target_col
                self.data = pd.read_csv(data_path, sep=None, engine='python')
                self.data.columns = self.data.columns.str.replace('.', '_')
                self.data = self.data.fillna(self.data.median(numeric_only=True))
            self.target = self.data.columns[int(target_col) - 1]
            self.exp = ClassificationExperiment() if task_type == 'classification' else RegressionExperiment()

        self.plots = {}

    def setup_pycaret(self):
        if self.exp is not None and hasattr(self.exp, 'is_setup') and self.exp.is_setup:
            LOG.info("Experiment already set up. Skipping PyCaret setup.")
            return
        LOG.info("Initializing PyCaret")
        setup_params = {
            'target': self.target,
            'session_id': 123,
            'html': True,
            'log_experiment': False,
            'system_log': False
        }
        self.exp.setup(self.data, **setup_params)

    def save_tree_importance(self):
        model = self.best_model or self.exp.get_config('best_model')
        processed_features = self.exp.get_config('X_transformed').columns

        # Try feature_importances_ or coef_ if available
        importances = None
        model_type = model.__class__.__name__
        self.tree_model_name = model_type  # Store the model name for reporting

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models, flatten coef_ and take abs (importance as magnitude)
            importances = abs(model.coef_).flatten()
        else:
            # Neither attribute exists; skip the plot
            LOG.warning(f"Model {model_type} does not have feature_importances_ or coef_ attribute. Skipping feature importance plot.")
            self.tree_model_name = None  # No plot generated
            return

        # Defensive: handle mismatch in number of features
        if len(importances) != len(processed_features):
            LOG.warning(
                f"Number of importances ({len(importances)}) does not match number of features ({len(processed_features)}). Skipping plot."
            )
            self.tree_model_name = None
            return

        feature_importances = pd.DataFrame({
            'Feature': processed_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_importances['Feature'],
            feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({model_type})')
        plot_path = os.path.join(
            self.output_dir,
            'tree_importance.png')
        plt.savefig(plot_path)
        plt.close()
        self.plots['tree_importance'] = plot_path

    def save_shap_values(self):
        model = self.best_model or self.exp.get_config('best_model')
        X_transformed = self.exp.get_config('X_transformed')
        tree_classes = (
            "LGBM", "XGB", "CatBoost", "RandomForest", "DecisionTree", "ExtraTrees", "HistGradientBoosting"
        )
        model_class_name = model.__class__.__name__
        self.shap_model_name = model_class_name

        # Ensure feature alignment
        if hasattr(model, "feature_name_"):
            used_features = model.feature_name_
        elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
            used_features = model.booster_.feature_name()
        elif hasattr(model, "feature_names_in_"):
            # scikit‐learn's standard attribute for the names of features used during fit
            used_features = list(model.feature_names_in_)
        else:
            used_features = X_transformed.columns

        if any(tc in model_class_name for tc in tree_classes):
            explainer = shap.TreeExplainer(model)
            X_shap = X_transformed[used_features]
            shap_values = explainer.shap_values(X_shap)
            plot_X = X_shap
            plot_title = f"SHAP Summary for {model_class_name} (TreeExplainer)"
        else:
            logging.warning(f"len(X_transformed) = {len(X_transformed)}")
            max_samples = 100
            n_samples = min(max_samples, len(X_transformed))
            sampled_X = X_transformed[used_features].sample(
                n=n_samples,
                replace=False,
                random_state=42
            )
            explainer = shap.KernelExplainer(model.predict, sampled_X)
            shap_values = explainer.shap_values(sampled_X)
            plot_X = sampled_X
            plot_title = f"SHAP Summary for {model_class_name} (KernelExplainer)"

        shap.summary_plot(shap_values, plot_X, show=False)
        plt.title(plot_title)
        plot_path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(plot_path)
        plt.close()
        self.plots["shap_summary"] = plot_path

    def generate_html_report(self):
        LOG.info("Generating HTML report")

        plots_html = ""
        for plot_name, plot_path in self.plots.items():
            # Special handling for tree importance: skip if no model name (not generated)
            if plot_name == 'tree_importance' and not getattr(self, 'tree_model_name', None):
                continue
            encoded_image = self.encode_image_to_base64(plot_path)
            if plot_name == 'tree_importance' and getattr(self, 'tree_model_name', None):
                section_title = f"Feature importance analysis from a trained {self.tree_model_name}"
            elif plot_name == 'shap_summary':
                section_title = f"SHAP Summary from a trained {getattr(self, 'shap_model_name', 'model')}"
            else:
                section_title = plot_name
            plots_html += f"""
            <div class="plot" id="{plot_name}">
                <h2>{section_title}</h2>
                <img src="data:image/png;base64,{encoded_image}" alt="{plot_name}">
            </div>
            """

        html_content = f"""
            <h1>PyCaret Feature Importance Report</h1>
            {plots_html}
        """

        return html_content

    def encode_image_to_base64(self, img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def run(self):
        if self.exp is None or not hasattr(self.exp, 'is_setup') or not self.exp.is_setup:
            self.setup_pycaret()
        self.save_tree_importance()
        self.save_shap_values()
        html_content = self.generate_html_report()
        return html_content
