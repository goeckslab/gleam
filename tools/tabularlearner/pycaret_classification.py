import logging
import types
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from base_model_trainer import BaseModelTrainer
from dashboard import generate_classifier_explainer_dashboard
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from utils import predict_proba

LOG = logging.getLogger(__name__)


def _apply_report_layout(fig: go.Figure) -> go.Figure:
    # Give the left side more space for y-axis title/ticks and let axes auto-reserve room
    fig.update_xaxes(automargin=True, title_standoff=12)
    fig.update_yaxes(automargin=True, title_standoff=12)
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(gridcolor="#e8e8e8")
    fig.update_yaxes(gridcolor="#e8e8e8")
    fig.update_layout(
        autosize=True,
        margin=dict(l=120, r=40, t=60, b=60),  # bump 'l' if you still see clipping
    )
    return fig


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
            "auc",
            "threshold",
            "pr",
            "error",
            "class_report",
            "learning",
            "calibration",
            "vc",
            "dimension",
            "manifold",
            "rfe",
            "feature",
            "feature_all",
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

        # Ensure predict_proba is available here too
        if not hasattr(self.best_model, "predict_proba"):
            self.best_model.predict_proba = types.MethodType(
                predict_proba, self.best_model
            )
            LOG.warning(
                f"The model {type(self.best_model).__name__} does not support `predict_proba`. Applying monkey patch."
            )

        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed
        explainer = ClassifierExplainer(self.best_model, X_test, y_test)

        # a dict to hold the raw Figure objects or callables
        self.explainer_plots: Dict[str, go.Figure] = {}

        # --- Threshold-aware overrides for CM / ROC / PR ---
        prob_thresh = getattr(self, "probability_threshold", None)

        # Only for binary classification and when threshold is provided
        if (prob_thresh is not None) and (not self.exp.is_multiclass):
            X = self.exp.X_test_transformed
            y = pd.Series(self.exp.y_test_transformed).reset_index(drop=True)

            # Get positive-class scores (robust defaults)
            classes = list(getattr(self.best_model, "classes_", [0, 1]))
            try:
                pos_idx = classes.index(1) if 1 in classes else 1
            except Exception:
                pos_idx = 1

            proba = self.best_model.predict_proba(X)
            y_scores = proba[:, pos_idx]

            # Derive label names consistently
            pos_label = classes[pos_idx] if len(classes) > pos_idx else 1
            neg_label = classes[1 - pos_idx] if len(classes) > 1 else 0

            # ---- Confusion Matrix @ threshold ----
            try:
                y_pred = np.where(y_scores >= prob_thresh, pos_label, neg_label)
                cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=[f"Pred {neg_label}", f"Pred {pos_label}"],
                        y=[f"True {neg_label}", f"True {pos_label}"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=False,
                    )
                )
                fig_cm.update_layout(
                    title=f"Confusion Matrix @ threshold={prob_thresh:.2f}",
                    xaxis_title="Predicted label",
                    yaxis_title="True label",
                )
                _apply_report_layout(fig_cm)
                self.explainer_plots["confusion_matrix"] = fig_cm
            except Exception as e:
                LOG.warning(
                    f"Threshold-aware confusion matrix failed; falling back: {e}"
                )

            # ---- ROC with threshold marker ----
            try:
                fpr, tpr, thr = roc_curve(y, y_scores)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_scatter(
                    x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"
                )
                if len(thr):
                    mask = np.isfinite(thr)
                    if mask.any():
                        idx_local = int(np.argmin(np.abs(thr[mask] - prob_thresh)))
                        idx = np.where(mask)[0][idx_local]
                        if 0 <= idx < len(fpr):
                            fig_roc.add_scatter(
                                x=[fpr[idx]],
                                y=[tpr[idx]],
                                mode="markers",
                                name=f"@ {prob_thresh:.2f}",
                                marker=dict(size=10),
                            )
                fig_roc.update_layout(
                    title=f"ROC Curve (marker at threshold={prob_thresh:.2f})",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                _apply_report_layout(fig_roc)
                self.explainer_plots["roc_auc"] = fig_roc
            except Exception as e:
                LOG.warning(f"Threshold marker on ROC failed; falling back: {e}")

            # ---- PR with threshold marker ----
            try:
                precision, recall, thr_pr = precision_recall_curve(y, y_scores)
                pr_auc = auc(recall, precision)
                fig_pr = go.Figure()
                fig_pr.add_scatter(
                    x=recall, y=precision, mode="lines", name=f"PR (AUC={pr_auc:.3f})"
                )
                if len(thr_pr):
                    idx_pr = int(np.argmin(np.abs(thr_pr - prob_thresh)))
                    # note: thr_pr has length = len(precision) - 1
                    idx_pr = max(0, min(idx_pr, len(recall) - 1))
                    fig_pr.add_scatter(
                        x=[recall[idx_pr]],
                        y=[precision[idx_pr]],
                        mode="markers",
                        name=f"@ {prob_thresh:.2f}",
                        marker=dict(size=10),
                    )
                fig_pr.update_layout(
                    title=f"Precision–Recall (marker at threshold={prob_thresh:.2f})",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                )
                _apply_report_layout(fig_pr)
                self.explainer_plots["pr_auc"] = fig_pr
            except Exception as e:
                LOG.warning(f"Threshold marker on PR failed; falling back: {e}")

        # these go into the Test tab (don't overwrite overrides)
        for key, fn in [
            ("roc_auc", explainer.plot_roc_auc),
            ("pr_auc", explainer.plot_pr_auc),
            ("lift_curve", explainer.plot_lift_curve),
            ("confusion_matrix", explainer.plot_confusion_matrix),
            ("threshold", explainer.plot_precision),  # percentage vs probability
            ("cumulative_precision", explainer.plot_cumulative_precision),
        ]:
            if key in self.explainer_plots:
                continue
            try:
                fig = fn()
                if fig is not None:
                    self.explainer_plots[key] = fig
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
                LOG.warning(
                    f"Skipping PDP for feature {feat!r}: not found in explainer data"
                )

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

