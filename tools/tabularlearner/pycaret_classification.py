import logging
import types
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from base_model_trainer import BaseModelTrainer
from dashboard import generate_classifier_explainer_dashboard
from pycaret.classification import ClassificationExperiment
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from utils import predict_proba

LOG = logging.getLogger(__name__)

MULTICLASS_UNAVAILABLE_PYCARET_PLOTS = {"threshold"}


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


def _should_skip_pycaret_plot(plot_name, exp):
    return (
        getattr(exp, "is_multiclass", False)
        and plot_name in MULTICLASS_UNAVAILABLE_PYCARET_PLOTS
    )


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

        if (
            not hasattr(self.best_model, "predict_proba")
            and not getattr(self.exp, "is_multiclass", False)
        ):
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
            "learning",
            "calibration",
            "vc",
            "rfe",
            "feature",
            "feature_all",
        ]
        for plot_name in plots:
            try:
                if _should_skip_pycaret_plot(plot_name, self.exp):
                    LOG.info(
                        "Skipping PyCaret %s plot for multiclass classification.",
                        plot_name,
                    )
                    continue
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
        if (
            not hasattr(self.best_model, "predict_proba")
            and not getattr(self.exp, "is_multiclass", False)
        ):
            self.best_model.predict_proba = types.MethodType(
                predict_proba, self.best_model
            )
            LOG.warning(
                f"The model {type(self.best_model).__name__} does not support `predict_proba`. Applying monkey patch."
            )

        X_test = self.exp.X_test_transformed.copy()
        y_test = pd.Series(self.exp.y_test_transformed).reset_index(drop=True)
        X_test, y_test = self._limit_explainer_data(
            X_test,
            y_test,
            context="ClassifierExplainer",
            cap_features=False,
        )
        label_encoder = None
        try:
            from pycaret.utils.generic import get_label_encoder

            label_encoder = get_label_encoder(self.exp.pipeline)
        except Exception as exc:
            LOG.debug("Could not load label encoder for explainer labels: %s", exc)
        explainer_labels = (
            list(label_encoder.classes_) if label_encoder is not None else None
        )
        explainer = None
        try:
            explainer = ClassifierExplainer(
                self.best_model,
                X_test,
                y_test,
                labels=explainer_labels,
            )
        except Exception as exc:
            LOG.warning(
                "Could not initialize ClassifierExplainer; "
                "continuing with custom classification plots only: %s",
                exc,
            )

        # a dict to hold the raw Figure objects or callables
        self.explainer_plots: Dict[str, go.Figure] = {}

        y_true, y_pred, label_values, y_scores = self._get_test_predictions()
        y_true_display = self._get_original_test_labels_for_display(y_true)
        y_pred_display = self._decode_labels_for_display(y_pred)
        label_values_display = pd.unique(
            pd.concat(
                [
                    pd.Series(y_true_display),
                    pd.Series(y_pred_display),
                ],
                ignore_index=True,
            )
        ).tolist()

        # — Classification report (Plotly table) —
        try:
            fig_report = self._build_classification_report_fig(
                y_true_display, y_pred_display, label_values_display
            )
            if fig_report is not None:
                self.explainer_plots["class_report"] = fig_report
        except Exception as e:
            LOG.warning(f"Could not generate Plotly classification report: {e}")

        # — Confusion matrix with actual labels —
        try:
            fig_cm = self._build_confusion_matrix_fig(
                y_true_display, y_pred_display, label_values_display
            )
            if fig_cm is not None:
                self.explainer_plots["confusion_matrix"] = fig_cm
        except Exception as e:
            LOG.warning(f"Could not generate Plotly confusion matrix: {e}")

        try:
            fig_dimension = self._build_dimension_reduction_fig()
            if fig_dimension is not None:
                self.explainer_plots["dimension"] = fig_dimension
        except Exception as e:
            LOG.warning(f"Could not generate custom dimensionality plot: {e}")

        try:
            fig_manifold = self._build_tsne_fig()
            if fig_manifold is not None:
                self.explainer_plots["manifold"] = fig_manifold
        except Exception as e:
            LOG.warning(f"Could not generate custom t-SNE plot: {e}")

        # --- Threshold-aware overrides for CM / ROC / PR ---
        prob_thresh = getattr(self, "probability_threshold", None)

        # Only for binary classification and when threshold is provided
        if (prob_thresh is not None) and (not self.exp.is_multiclass):
            # ---- ROC with threshold marker ----
            try:
                if y_scores is None:
                    raise ValueError("Predicted probabilities unavailable")
                fpr, tpr, thr = roc_curve(y_true, y_scores)
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
                fig_pr = self._build_precision_recall_fig(y_true, y_scores)
                if fig_pr is None:
                    raise ValueError("Predicted probabilities unavailable")
                self.explainer_plots["pr_auc"] = fig_pr
            except Exception as e:
                LOG.warning(f"Threshold marker on PR failed; falling back: {e}")

        if "pr_auc" not in self.explainer_plots:
            try:
                fig_pr = self._build_precision_recall_fig(y_true, y_scores)
                if fig_pr is not None:
                    self.explainer_plots["pr_auc"] = fig_pr
            except Exception as e:
                LOG.warning(f"Could not generate custom PR curve: {e}")

        if explainer is None:
            return

        # these go into the Test tab (don't overwrite overrides)
        for key, fn in [
            ("roc_auc", explainer.plot_roc_auc),
            ("pr_auc", explainer.plot_pr_auc),
            ("lift_curve", explainer.plot_lift_curve),
            ("confusion_matrix", explainer.plot_confusion_matrix),
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

        if self._explainer_feature_count_exceeds_cap():
            LOG.info(
                "Skipping ExplainerDashboard SHAP/permutation importance because "
                "the transformed test set has %s features; custom SHAP remains capped "
                "to top %s features.",
                self.explainer_scope.total_features,
                self.explainer_scope.feature_cap,
            )
            self.explainer_dashboard_importance_skipped = True
        else:
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
        for feat in self.plot_feature_names:
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

    def _get_test_predictions(self):
        """
        Return y_true, y_pred, label list, and (optionally) positive-class
        probabilities when available. Ensures predictions respect the optional
        probability threshold for binary tasks.
        """
        y_true = pd.Series(self.exp.y_test_transformed).reset_index(drop=True)
        X_test = self.exp.X_test_transformed
        prob_thresh = getattr(self, "probability_threshold", None)

        y_scores = None
        try:
            proba = self.best_model.predict_proba(X_test)
            y_scores = proba
        except Exception:
            LOG.debug("predict_proba unavailable for test predictions.")

        try:
            if (
                prob_thresh is not None
                and not self.exp.is_multiclass
                and y_scores is not None
                and y_scores.ndim == 2
                and y_scores.shape[1] > 1
            ):
                classes = list(getattr(self.best_model, "classes_", []))
                try:
                    pos_idx = classes.index(1) if 1 in classes else 1
                except Exception:
                    pos_idx = 1
                neg_idx = 1 - pos_idx if y_scores.shape[1] > 1 else 0
                pos_label = classes[pos_idx] if len(classes) > pos_idx else 1
                neg_label = classes[neg_idx] if len(classes) > neg_idx else 0
                y_pred = np.where(y_scores[:, pos_idx] >= prob_thresh, pos_label, neg_label)
                y_scores = y_scores[:, pos_idx]
            else:
                y_pred = self.best_model.predict(X_test)
        except Exception as exc:
            LOG.warning("Falling back to raw predict for test predictions: %s", exc)
            y_pred = self.best_model.predict(X_test)

        y_pred = pd.Series(y_pred).reset_index(drop=True)
        if y_scores is not None:
            y_scores = np.asarray(y_scores)
            if y_scores.ndim > 1 and y_scores.shape[1] == 1:
                y_scores = y_scores.ravel()
            if self.exp.is_multiclass and y_scores.ndim > 1:
                # Avoid passing multiclass score matrices to ROC/PR utilities
                y_scores = None
        label_values = pd.unique(pd.concat([y_true, y_pred], ignore_index=True))
        return y_true, y_pred, label_values.tolist(), y_scores

    def _decode_labels_for_display(self, values):
        """Map transformed class ids back to the original target labels for plots."""
        return self._decode_class_labels_for_display(values)

    def _get_original_test_labels_for_display(self, fallback):
        """Return original test labels for report plots when PyCaret exposes them."""
        candidates = []
        if self.test_data is not None and self.target in self.test_data.columns:
            candidates.append(self.test_data[self.target])
        for key in ("y_test", "y_test_transformed"):
            try:
                candidates.append(self.exp.get_config(key))
            except Exception:
                candidates.append(getattr(self.exp, key, None))

        fallback_len = len(fallback) if fallback is not None else None
        for candidate in candidates:
            if candidate is None:
                continue
            series = pd.Series(candidate).reset_index(drop=True)
            if fallback_len is not None and len(series) != fallback_len:
                continue
            return self._decode_labels_for_display(series)

        return self._decode_labels_for_display(fallback)

    def _get_embedding_data_for_display(self, max_rows=None):
        X = self._get_exp_config(["X_train_transformed", "X_train"])
        y = self._get_exp_config(["y_train", "y_train_transformed"])
        if X is None or y is None:
            return None, None

        X_df = pd.DataFrame(X).reset_index(drop=True)
        X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            return None, None
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

        y_series = pd.Series(y).reset_index(drop=True)
        row_count = min(len(X_df), len(y_series))
        X_df = X_df.iloc[:row_count].reset_index(drop=True)
        y_series = y_series.iloc[:row_count].reset_index(drop=True)
        y_series = self._decode_labels_for_display(y_series)

        if max_rows is not None and len(X_df) > max_rows:
            sample_idx = (
                X_df.sample(n=max_rows, random_state=self.random_seed)
                .sort_index()
                .index
            )
            X_df = X_df.loc[sample_idx].reset_index(drop=True)
            y_series = y_series.loc[sample_idx].reset_index(drop=True)

        return X_df, y_series

    def _get_exp_config(self, keys):
        for key in keys:
            try:
                value = self.exp.get_config(key)
            except Exception:
                value = getattr(self.exp, key, None)
            if value is not None:
                return value
        return None

    def _plot_labeled_scatter(self, fig, x, y, labels, hover_text=None):
        labels = pd.Series(labels).reset_index(drop=True)

        def _label_sort_key(lbl):
            try:
                return (0, float(lbl))
            except Exception:
                return (1, str(lbl))

        for label in sorted(pd.unique(labels), key=_label_sort_key):
            mask = labels == label
            if pd.isna(label):
                mask = labels.isna()
                label_name = "NaN"
            else:
                label_name = str(label)
            customdata = None
            if hover_text is not None:
                customdata = np.asarray(hover_text)[mask.to_numpy()]
            fig.add_scatter(
                x=np.asarray(x)[mask.to_numpy()],
                y=np.asarray(y)[mask.to_numpy()],
                mode="markers",
                name=label_name,
                customdata=customdata,
                marker=dict(size=7, opacity=0.72),
                hovertemplate=(
                    "Label=%{fullData.name}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<extra></extra>"
                    if hover_text is None
                    else "%{customdata}<br>Label=%{fullData.name}<extra></extra>"
                ),
            )

    def _build_dimension_reduction_fig(self):
        X_df, labels = self._get_embedding_data_for_display(max_rows=1500)
        if X_df is None or len(X_df) < 2:
            return None

        variances = X_df.var(axis=0).sort_values(ascending=False)
        selected_cols = list(variances.head(min(5, len(variances))).index)
        X_selected = X_df[selected_cols].copy()
        denom = X_selected.max(axis=0) - X_selected.min(axis=0)
        denom = denom.replace(0, 1)
        X_norm = (X_selected - X_selected.min(axis=0)) / denom

        feature_count = len(selected_cols)
        angles = np.linspace(0, 2 * np.pi, feature_count, endpoint=False)
        anchors = np.column_stack((np.cos(angles), np.sin(angles)))
        weights = X_norm.to_numpy(dtype=float)
        weight_sum = weights.sum(axis=1)
        weight_sum[weight_sum == 0] = 1
        coords = weights.dot(anchors) / weight_sum[:, None]

        fig = go.Figure()
        self._plot_labeled_scatter(fig, coords[:, 0], coords[:, 1], labels)
        circle = np.linspace(0, 2 * np.pi, 180)
        fig.add_scatter(
            x=np.cos(circle),
            y=np.sin(circle),
            mode="lines",
            line=dict(color="#9a9a9a", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
        for i, (x_coord, y_coord) in enumerate(anchors):
            fig.add_scatter(
                x=[x_coord],
                y=[y_coord],
                mode="markers+text",
                marker=dict(color="#777777", size=8),
                text=[str(selected_cols[i])],
                textposition="top center",
                showlegend=False,
                hovertemplate=f"Feature={selected_cols[i]}<extra></extra>",
            )
        fig.update_layout(
            title=f"RadViz for {feature_count} Features",
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
            legend_title_text=f"Label ({self.target})",
        )
        _apply_report_layout(fig)
        return fig

    def _build_tsne_fig(self):
        X_df, labels = self._get_embedding_data_for_display(max_rows=1500)
        if X_df is None or len(X_df) < 4:
            return None

        perplexity = min(30, max(2, (len(X_df) - 1) // 3))
        pca_components = min(50, X_df.shape[1], len(X_df) - 1)
        X_values = X_df.to_numpy(dtype=float)
        if pca_components >= 2 and X_df.shape[1] > pca_components:
            X_values = PCA(
                n_components=pca_components,
                random_state=self.random_seed,
            ).fit_transform(X_values)

        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=self.random_seed,
        ).fit_transform(X_values)

        fig = go.Figure()
        self._plot_labeled_scatter(
            fig,
            embedding[:, 0],
            embedding[:, 1],
            labels,
        )
        fig.update_layout(
            title=f"t-SNE Manifold for {len(X_df)} Samples",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            legend_title_text=f"Label ({self.target})",
        )
        _apply_report_layout(fig)
        return fig

    def _threshold_suffix(self) -> str:
        """
        Build a suffix like ' (threshold=0.50)' for binary tasks; omit for
        multiclass where thresholds are not applied.
        """
        if getattr(self, "task_type", None) != "classification":
            return ""
        if getattr(self.exp, "is_multiclass", False):
            return ""
        prob_thresh = getattr(self, "probability_threshold", None)
        if prob_thresh is None:
            return " (threshold=0.50)"
        try:
            return f" (threshold={float(prob_thresh):.2f})"
        except Exception:
            return f" (threshold={prob_thresh})"

    def _build_precision_recall_fig(self, y_true, y_scores):
        """
        Build a binary precision-recall curve with recall on X and precision on Y.
        Returns None when class probabilities are unavailable or not binary.
        """
        if y_scores is None or getattr(self.exp, "is_multiclass", False):
            return None

        y_scores = np.asarray(y_scores)
        if y_scores.ndim != 1:
            return None

        precision, recall, thr_pr = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        fig_pr = go.Figure()
        fig_pr.add_scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR (AUC={pr_auc:.3f})",
        )

        prob_thresh = getattr(self, "probability_threshold", None)
        if prob_thresh is not None and len(thr_pr):
            idx_pr = int(np.argmin(np.abs(thr_pr - prob_thresh)))
            idx_pr = max(0, min(idx_pr, len(recall) - 1))
            fig_pr.add_scatter(
                x=[recall[idx_pr]],
                y=[precision[idx_pr]],
                mode="markers",
                name=f"@ {prob_thresh:.2f}",
                marker=dict(size=10),
            )

        fig_pr.update_layout(
            title=f"Precision-Recall Curve{self._threshold_suffix()}",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        _apply_report_layout(fig_pr)
        return fig_pr

    def _build_confusion_matrix_fig(self, y_true, y_pred, labels):
        def _label_sort_key(lbl):
            try:
                return (0, float(lbl))
            except Exception:
                return (1, str(lbl))

        ordered_labels = sorted(labels, key=_label_sort_key)
        cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)
        label_names = [str(lbl) for lbl in ordered_labels]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Pred {lbl}" for lbl in label_names],
                y=[f"True {lbl}" for lbl in label_names],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=f"Confusion Matrix{self._threshold_suffix()}",
            xaxis_title=f"Predicted label ({self.target})",
            yaxis_title=f"True label ({self.target})",
        )
        fig_cm.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=[f"Pred {lbl}" for lbl in label_names],
        )
        fig_cm.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=[f"True {lbl}" for lbl in label_names],
            autorange="reversed",
        )
        _apply_report_layout(fig_cm)
        return fig_cm

    def _build_classification_report_fig(self, y_true, y_pred, labels):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        mcc_scores = []
        for lbl in labels:
            y_true_bin = (y_true == lbl).astype(int)
            y_pred_bin = (y_pred == lbl).astype(int)
            try:
                mcc_val = matthews_corrcoef(y_true_bin, y_pred_bin)
            except Exception:
                mcc_val = 0.0
            mcc_scores.append(mcc_val)

        label_names = [str(lbl) for lbl in labels]
        metrics = ["precision", "recall", "f1", "support"]

        max_support = float(max(support) if len(support) else 0)
        z_rows = []
        text_rows = []
        for i, lbl in enumerate(label_names):
            norm_support = (support[i] / max_support) if max_support else 0.0
            z_rows.append(
                [
                    precision[i],
                    recall[i],
                    f1[i],
                    norm_support,
                ]
            )
            text_rows.append(
                [
                    f"{precision[i]:.3f}",
                    f"{recall[i]:.3f}",
                    f"{f1[i]:.3f}",
                    f"{int(support[i])}",
                ]
            )

        fig = go.Figure(
            data=go.Heatmap(
                z=z_rows,
                x=metrics,
                y=label_names,
                colorscale="YlOrRd",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Scale"),
                text=text_rows,
                texttemplate="%{text}",
                hovertemplate="Label=%{y}<br>Metric=%{x}<br>Value=%{text}<extra></extra>",
            )
        )
        fig.update_yaxes(
            title_text=f"Label ({self.target})",
            autorange="reversed",
            type="category",
            tickmode="array",
            tickvals=label_names,
            ticktext=label_names,
            showgrid=False,
        )
        fig.update_xaxes(title_text="", tickangle=45)
        fig.update_layout(
            title=f"Per-Class Metrics{self._threshold_suffix()}",
            margin=dict(l=70, r=60, t=70, b=80),
        )
        _apply_report_layout(fig)
        return fig
