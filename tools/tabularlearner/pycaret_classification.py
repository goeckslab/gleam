import logging
import types
from typing import Dict

from base_model_trainer import BaseModelTrainer
from dashboard import generate_classifier_explainer_dashboard
from plotly.graph_objects import Figure
from pycaret.classification import ClassificationExperiment
from utils import predict_proba

LOG = logging.getLogger(__name__)


class ClassificationModelTrainer(BaseModelTrainer):
    def __init__(
        self,
        input_file,
        target_col,
        output_dir,
        task_type,
        random_seed,
        test_file=None,
        **kwargs,
    ):
        super().__init__(
            input_file,
            target_col,
            output_dir,
            task_type,
            random_seed,
            test_file,
            **kwargs,
        )
        self.exp = ClassificationExperiment()

    def save_dashboard(self):
        LOG.info("Saving explainer dashboard")
        dashboard = generate_classifier_explainer_dashboard(self.exp, self.best_model)
        dashboard.save_html("dashboard.html")

    def generate_plots(self):
        LOG.info("Generating and saving plots")

        if not hasattr(self.best_model, "predict_proba"):
            self.best_model.predict_proba = types.MethodType(
                predict_proba, self.best_model
            )
            LOG.warning(
                f"The model {type(self.best_model).__name__} does not support `predict_proba`. Applying monkey patch."
            )

        plots = [
            'confusion_matrix',
            'auc',
            'threshold',
            'pr',
            'error',
            'class_report',
            'learning',
            'calibration',
            'vc',
            'dimension',
            'manifold',
            'rfe',
            'feature',
            'feature_all',
        ]
        for plot_name in plots:
            try:
                if plot_name == "threshold":
                    plot_path = self.exp.plot_model(
                        self.best_model,
                        plot=plot_name,
                        save=True,
                        plot_kwargs={"binary": True, "percentage": True},
                    )
                    self.plots[plot_name] = plot_path
                elif plot_name == "auc" and not self.exp.is_multiclass:
                    plot_path = self.exp.plot_model(
                        self.best_model,
                        plot=plot_name,
                        save=True,
                        plot_kwargs={
                            "micro": False,
                            "macro": False,
                            "per_class": False,
                            "binary": True,
                        },
                    )
                    self.plots[plot_name] = plot_path
                else:
                    plot_path = self.exp.plot_model(
                        self.best_model, plot=plot_name, save=True
                    )
                    self.plots[plot_name] = plot_path
            except Exception as e:
                LOG.error(f"Error generating plot {plot_name}: {e}")
                continue

    def generate_plots_explainer(self):
        from explainerdashboard import ClassifierExplainer

        LOG.info("Generating explainer plots")

        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed
        explainer = ClassifierExplainer(self.best_model, X_test, y_test)

        # a dict to hold the raw Figure objects or callables
        self.explainer_plots: Dict[str, Figure] = {}

        # these go into the Test tab
        for key, fn in [
            ("roc_auc", explainer.plot_roc_auc),
            ("pr_auc", explainer.plot_pr_auc),
            ("lift_curve", explainer.plot_lift_curve),
            ("confusion_matrix", explainer.plot_confusion_matrix),
            ("threshold", explainer.plot_precision),  # Percentage vs probability
            ("cumulative_precision", explainer.plot_cumulative_precision),
        ]:
            try:
                self.explainer_plots[key] = fn()
            except Exception as e:
                LOG.error(f"Error generating explainer plot {key}: {e}")

        # mean SHAP importances
        try:
            self.explainer_plots["shap_mean"] = explainer.plot_importances()
        except Exception as e:
            LOG.warning(f"Could not generate shap_mean: {e}")

        # permutation importances
        try:
            self.explainer_plots["shap_perm"] = lambda: explainer.plot_importances(
                kind="permutation"
            )
        except Exception as e:
            LOG.warning(f"Could not generate shap_perm: {e}")

        # PDPs for each feature (appended last)
        valid_feats = []
        for feat in self.features_name:
            if feat in explainer.X.columns or feat in explainer.onehot_cols:
                valid_feats.append(feat)
            else:
                LOG.warning(f"Skipping PDP for feature {feat!r}: not found in explainer data")

        for feat in valid_feats:
            # wrap each PDP call to catch any unexpected AssertionErrors
            def make_pdp_plotter(f):
                def _plot():
                    try:
                        return explainer.plot_pdp(f)
                    except AssertionError as ae:
                        LOG.warning(f"PDP AssertionError for {f!r}: {ae}")
                        return None
                    except Exception as e:
                        LOG.error(f"Unexpected error plotting PDP for {f!r}: {e}")
                        return None
                return _plot

            self.explainer_plots[f"pdp__{feat}"] = make_pdp_plotter(feat)
