import logging

from base_model_trainer import BaseModelTrainer
from dashboard import generate_regression_explainer_dashboard
from pycaret.regression import RegressionExperiment

LOG = logging.getLogger(__name__)


class RegressionModelTrainer(BaseModelTrainer):
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
        # The BaseModelTrainer.setup_pycaret will set self.exp appropriately
        # But we reassign here for clarity
        self.exp = RegressionExperiment()

    def save_dashboard(self):
        LOG.info("Saving explainer dashboard")
        dashboard = generate_regression_explainer_dashboard(self.exp, self.best_model)
        dashboard.save_html("dashboard.html")

    def generate_plots(self):
        LOG.info("Generating and saving plots")
        plots = [
            "residuals",
            "error",
            "cooks",
            "learning",
            "vc",
            "manifold",
            "rfe",
            "feature",
            "feature_all",
        ]
        for plot_name in plots:
            try:
                plot_path = self.exp.plot_model(
                    self.best_model, plot=plot_name, save=True
                )
                self.plots[plot_name] = plot_path
            except Exception as e:
                LOG.error(f"Error generating plot {plot_name}: {e}")
                continue

    def generate_plots_explainer(self):
        LOG.info("Generating and saving plots from explainer")

        from explainerdashboard import RegressionExplainer

        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed

        try:
            explainer = RegressionExplainer(self.best_model, X_test, y_test)
        except Exception as e:
            LOG.error(f"Error creating explainer: {e}")
            return

        # 1) SHAP mean impact (average absolute SHAP values)
        try:
            fig_shap_mean = explainer.plot_importances()
            self.explainer_plots["shap_mean"] = fig_shap_mean
        except Exception as e:
            LOG.error(f"Error generating SHAP mean importance: {e}")

        # 2) SHAP permutation importance
        try:
            fig_shap_perm = explainer.plot_importances_permutation(kind="permutation")
            self.explainer_plots["shap_perm"] = fig_shap_perm
        except Exception as e:
            LOG.error(f"Error generating SHAP permutation importance: {e}")

        # 3) Partial Dependence Plots (PDPs) per feature
        for feature in self.features_name:
            try:
                fig_pdp = explainer.plot_pdp(feature)
                self.explainer_plots[f"pdp__{feature}"] = fig_pdp
            except Exception as e:
                LOG.error(f"Error generating PDP for {feature}: {e}")

        # 4) Predicted vs Actual charges (optional)
        try:
            fig_pred_actual = explainer.plot_predicted_vs_actual()
            self.explainer_plots["predicted_vs_actual"] = fig_pred_actual
        except Exception as e:
            LOG.error(f"Error generating Predicted vs Actual plot: {e}")

        # 5) Global residuals distribution
        try:
            fig_residuals = explainer.plot_residuals()
            self.explainer_plots["residuals"] = fig_residuals
        except Exception as e:
            LOG.error(f"Error generating Residuals plot: {e}")

        # 6) Residuals vs each feature
        for feature in self.features_name:
            try:
                fig_res_vs_feat = explainer.plot_residuals_vs_feature(feature)
                self.explainer_plots[f"residuals_vs_feature__{feature}"] = (
                    fig_res_vs_feat
                )
            except Exception as e:
                LOG.error(f"Error generating Residuals vs {feature}: {e}")
