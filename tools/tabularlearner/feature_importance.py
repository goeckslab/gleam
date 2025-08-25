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
        best_model=None,
    ):

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
            self.exp = (
                ClassificationExperiment()
                if task_type == "classification"
                else RegressionExperiment()
            )

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
            'system_log': False,
        }
        self.exp.setup(self.data, **setup_params)

    def save_tree_importance(self):
        model = self.best_model or self.exp.get_config('best_model')
        processed_features = self.exp.get_config('X_transformed').columns

        # Try feature_importances_ or coef_ if available
        importances = None
        model_type = model.__class__.__name__
        self.tree_model_name = model_type  # Store the model name for reporting

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, flatten coef_ and take abs (importance as magnitude)
            importances = abs(model.coef_).flatten()
        else:
            # Neither attribute exists; skip the plot
            LOG.warning(
                f"Model {model_type} does not have feature_importances_ or coef_ attribute. Skipping feature importance plot."
            )
            self.tree_model_name = None  # No plot generated
            return

        # Defensive: handle mismatch in number of features
        if len(importances) != len(processed_features):
            LOG.warning(
                f"Number of importances ({len(importances)}) does not match number of features ({len(processed_features)}). Skipping plot."
            )
            self.tree_model_name = None
            return

        feature_importances = pd.DataFrame(
            {'Feature': processed_features, 'Importance': importances}
        ).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({model_type})')
        plot_path = os.path.join(self.output_dir, 'tree_importance.png')
        plt.savefig(plot_path)
        plt.close()
        self.plots['tree_importance'] = plot_path

    def save_shap_values(self):

        model = self.best_model or self.exp.get_config("best_model")

        X_data = None
        for key in ("X_test_transformed", "X_train_transformed"):
            try:
                X_data = self.exp.get_config(key)
                break
            except KeyError:
                continue
        if X_data is None:
            raise RuntimeError(
                "Could not find 'X_test_transformed' or 'X_train_transformed' in the experiment. "
                "Make sure PyCaret setup/compare_models was run with feature_selection=True."
            )

        try:
            used_features = model.booster_.feature_name()
        except Exception:
            used_features = getattr(model, "feature_names_in_", X_data.columns.tolist())
        X_data = X_data[used_features]

        max_bg = min(len(X_data), 100)
        bg = X_data.sample(max_bg, random_state=42)

        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

        try:
            explainer = shap.Explainer(predict_fn, bg)
            self.shap_model_name = explainer.__class__.__name__

            shap_values = explainer(X_data)
        except Exception as e:
            LOG.error(f"SHAP computation failed: {e}")
            self.shap_model_name = None
            return

        output_names = getattr(shap_values, "output_names", None)
        if output_names is None and hasattr(model, "classes_"):
            output_names = list(model.classes_)
        if output_names is None:
            n_out = shap_values.values.shape[-1]
            output_names = list(map(str, range(n_out)))

        values = shap_values.values
        if values.ndim == 3:
            for j, name in enumerate(output_names):
                safe = name.replace(" ", "_").replace("/", "_")
                out_path = os.path.join(self.output_dir, f"shap_summary_{safe}.png")
                plt.figure()
                shap.plots.beeswarm(shap_values[..., j], show=False)
                plt.title(f"SHAP for {model.__class__.__name__} ⇒ {name}")
                plt.savefig(out_path)
                plt.close()
                self.plots[f"shap_summary_{safe}"] = out_path
        else:
            plt.figure()
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f"SHAP Summary for {model.__class__.__name__}")
            out_path = os.path.join(self.output_dir, "shap_summary.png")
            plt.savefig(out_path)
            plt.close()
            self.plots["shap_summary"] = out_path

    def generate_html_report(self):
        LOG.info("Generating HTML report")

        plots_html = ""
        for plot_name, plot_path in self.plots.items():
            # Special handling for tree importance: skip if no model name (not generated)
            if plot_name == 'tree_importance' and not getattr(
                self, 'tree_model_name', None
            ):
                continue
            encoded_image = self.encode_image_to_base64(plot_path)
            if plot_name == 'tree_importance' and getattr(
                self, 'tree_model_name', None
            ):
                section_title = (
                    f"Feature importance analysis from a trained {self.tree_model_name}"
                )
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
            {plots_html}
        """

        return html_content

    def encode_image_to_base64(self, img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def run(self):
        if (
            self.exp is None
            or not hasattr(self.exp, 'is_setup')
            or not self.exp.is_setup
        ):
            self.setup_pycaret()
        self.save_tree_importance()
        self.save_shap_values()
        html_content = self.generate_html_report()
        return html_content
