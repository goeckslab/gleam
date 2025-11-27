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
        max_plot_features=None,
        processed_data=None,
        max_shap_rows=None,
    ):
        self.task_type = task_type
        self.output_dir = output_dir
        self.exp = exp
        self.best_model = best_model
        self._skip_messages = []
        self.shap_total_features = None
        self.shap_used_features = None
        if isinstance(max_plot_features, int) and max_plot_features > 0:
            self.max_plot_features = max_plot_features
        elif max_plot_features is None:
            self.max_plot_features = 30
        else:
            self.max_plot_features = None

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
                self.data = pd.read_csv(data_path, sep=None, engine="python")
                self.data.columns = self.data.columns.str.replace(".", "_")
                self.data = self.data.fillna(self.data.median(numeric_only=True))
            self.target = self.data.columns[int(target_col) - 1]
            self.exp = (
                ClassificationExperiment()
                if task_type == "classification"
                else RegressionExperiment()
            )
        if processed_data is not None:
            self.data = processed_data

        self.plots = {}
        self.max_shap_rows = max_shap_rows

    def _get_feature_names_from_model(self, model):
        """Best-effort extraction of feature names seen by the estimator."""
        if model is None:
            return None

        candidates = [model]
        if hasattr(model, "named_steps"):
            candidates.extend(model.named_steps.values())
        elif hasattr(model, "steps"):
            candidates.extend(step for _, step in model.steps)

        for candidate in candidates:
            names = getattr(candidate, "feature_names_in_", None)
            if names is not None:
                return list(names)
        return None

    def _get_transformed_frame(self, model=None, prefer_test=True):
        """Return a DataFrame that mirrors the matrix fed to the estimator."""
        key_order = ["X_test_transformed", "X_train_transformed"]
        if not prefer_test:
            key_order.reverse()
        key_order.append("X_transformed")

        feature_names = self._get_feature_names_from_model(model)
        for key in key_order:
            try:
                frame = self.exp.get_config(key)
            except KeyError:
                continue
            if frame is None:
                continue
            if isinstance(frame, pd.DataFrame):
                return frame.copy()
            try:
                n_features = frame.shape[1]
            except Exception:
                continue
            if feature_names and len(feature_names) == n_features:
                return pd.DataFrame(frame, columns=feature_names)
            # Fallback to positional names so downstream logic still works
            return pd.DataFrame(frame, columns=[f"f{i}" for i in range(n_features)])
        return None

    def setup_pycaret(self):
        if self.exp is not None and hasattr(self.exp, "is_setup") and self.exp.is_setup:
            LOG.info("Experiment already set up. Skipping PyCaret setup.")
            return
        LOG.info("Initializing PyCaret")
        setup_params = {
            "target": self.target,
            "session_id": 123,
            "html": True,
            "log_experiment": False,
            "system_log": False,
        }
        self.exp.setup(self.data, **setup_params)

    def save_tree_importance(self):
        model = self.best_model or self.exp.get_config("best_model")
        processed_frame = self._get_transformed_frame(model, prefer_test=False)
        if processed_frame is None:
            LOG.warning(
                "Unable to determine transformed feature names; skipping tree importance plot."
            )
            self.tree_model_name = None
            return
        processed_features = list(processed_frame.columns)

        importances = None
        model_type = model.__class__.__name__
        self.tree_model_name = model_type

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = abs(model.coef_).flatten()
        else:
            LOG.warning(
                f"Model {model_type} does not have feature_importances_ or coef_. Skipping tree importance."
            )
            self.tree_model_name = None
            return

        if len(importances) != len(processed_features):
            model_feature_names = self._get_feature_names_from_model(model)
            if model_feature_names and len(model_feature_names) == len(importances):
                processed_features = model_feature_names
            else:
                LOG.warning(
                    "Importances (%s) != features (%s). Skipping tree importance.",
                    len(importances),
                    len(processed_features),
                )
                self.tree_model_name = None
                return

        feature_importances = pd.DataFrame(
            {"Feature": processed_features, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)
        cap = (
            min(self.max_plot_features, len(feature_importances))
            if self.max_plot_features is not None
            else len(feature_importances)
        )
        plot_importances = feature_importances.head(cap)
        if cap < len(feature_importances):
            LOG.info(
                "Tree importance plot limited to top %s of %s features",
                cap,
                len(feature_importances),
            )
        plt.figure(figsize=(10, 6))
        plt.barh(
            plot_importances["Feature"],
            plot_importances["Importance"],
        )
        plt.xlabel("Importance")
        plt.title(f"Feature Importance ({model_type}) (top {cap})")
        plot_path = os.path.join(self.output_dir, "tree_importance.png")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        self.plots["tree_importance"] = plot_path

    def save_shap_values(self, max_samples=None, max_display=None, max_features=None):
        model = self.best_model or self.exp.get_config("best_model")

        X_data = self._get_transformed_frame(model)
        if X_data is None:
            raise RuntimeError("No transformed dataset found for SHAP.")

        n_rows, n_features = X_data.shape
        self.shap_total_features = n_features
        feature_cap = (
            min(self.max_plot_features, n_features)
            if self.max_plot_features is not None
            else n_features
        )
        if max_features is None:
            max_features = feature_cap
        else:
            max_features = min(max_features, feature_cap)
        display_features = list(X_data.columns)

        try:
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(
                    model.feature_importances_, index=X_data.columns
                )
                top_features = importances.nlargest(max_features).index
            elif hasattr(model, "coef_"):
                coef = abs(model.coef_).flatten()
                importances = pd.Series(coef, index=X_data.columns)
                top_features = importances.nlargest(max_features).index
            else:
                variances = X_data.var()
                top_features = variances.nlargest(max_features).index

            candidate_features = list(top_features)
            missing = [f for f in candidate_features if f not in X_data.columns]
            display_features = [f for f in candidate_features if f in X_data.columns]
            if missing:
                LOG.warning(
                    "Dropping %s transformed feature(s) not present in SHAP frame: %s",
                    len(missing),
                    missing[:5],
                )
            if display_features and len(display_features) < n_features:
                LOG.info(
                    "Restricting SHAP display to top %s of %s features",
                    len(display_features),
                    n_features,
                )
            elif not display_features:
                display_features = list(X_data.columns)
        except Exception as e:
            LOG.warning(
                f"Feature limiting failed: {e}. Using all {n_features} features."
            )
            display_features = list(X_data.columns)

        self.shap_used_features = len(display_features)

        # Apply the column restriction so SHAP only runs on the selected features.
        if display_features:
            X_data = X_data[display_features]
            n_rows, n_features = X_data.shape

        # --- Adaptive row subsampling ---
        if max_samples is None:
            if n_rows <= 500:
                max_samples = n_rows
            elif n_rows <= 5000:
                max_samples = 500
            else:
                max_samples = min(1000, int(n_rows * 0.1))

        if self.max_shap_rows is not None:
            max_samples = min(max_samples, self.max_shap_rows)

        if n_rows > max_samples:
            LOG.info(f"Subsampling SHAP rows: {max_samples} of {n_rows}")
            X_data = X_data.sample(max_samples, random_state=42)

        # --- Adaptive feature display ---
        display_cap = (
            min(self.max_plot_features, len(display_features))
            if self.max_plot_features is not None
            else len(display_features)
        )
        if max_display is None:
            max_display = display_cap
        else:
            max_display = min(max_display, display_cap)
        if not display_features:
            display_features = list(X_data.columns)
            max_display = len(display_features)

        # Background set
        bg = X_data.sample(min(len(X_data), 100), random_state=42)
        predict_fn = (
            model.predict_proba if hasattr(model, "predict_proba") else model.predict
        )

        # Optimized explainer
        explainer = None
        explainer_label = None
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(
                model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
            )
            explainer_label = "tree_path_dependent"
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, bg)
            explainer_label = "linear"
        else:
            explainer = shap.Explainer(predict_fn, bg)
            explainer_label = explainer.__class__.__name__

        try:
            shap_values = explainer(X_data)
            self.shap_model_name = explainer.__class__.__name__
        except Exception as e:
            error_message = str(e)
            needs_tree_fallback = (
                hasattr(model, "feature_importances_")
                and "does not cover all the leaves" in error_message.lower()
            )
            feature_name_mismatch = "feature names should match" in error_message.lower()
            if needs_tree_fallback:
                LOG.warning(
                    "SHAP computation failed using '%s' perturbation (%s). "
                    "Retrying with interventional perturbation.",
                    explainer_label,
                    error_message,
                )
                try:
                    explainer = shap.TreeExplainer(
                        model,
                        bg,
                        feature_perturbation="interventional",
                        n_jobs=-1,
                    )
                    shap_values = explainer(X_data)
                    self.shap_model_name = (
                        f"{explainer.__class__.__name__} (interventional)"
                    )
                except Exception as retry_exc:
                    LOG.error(
                        "SHAP computation failed even after fallback: %s",
                        retry_exc,
                    )
                    self.shap_model_name = None
                    return
            elif feature_name_mismatch:
                LOG.warning(
                    "SHAP computation failed due to feature-name mismatch (%s). "
                    "Falling back to model-agnostic SHAP explainer.",
                    error_message,
                )
                try:
                    agnostic_explainer = shap.Explainer(predict_fn, bg)
                    shap_values = agnostic_explainer(X_data)
                    self.shap_model_name = (
                        f"{agnostic_explainer.__class__.__name__} (fallback)"
                    )
                except Exception as fallback_exc:
                    LOG.error(
                        "Model-agnostic SHAP fallback also failed: %s",
                        fallback_exc,
                    )
                    self.shap_model_name = None
                    return
            else:
                LOG.error(f"SHAP computation failed: {e}")
                self.shap_model_name = None
                return

        def _limit_explanation_features(explanation):
            if len(display_features) >= n_features:
                return explanation
            try:
                limited = explanation[:, display_features]
                LOG.info(
                    "SHAP explanation trimmed to %s display features.",
                    len(display_features),
                )
                return limited
            except Exception as exc:
                LOG.warning(
                    "Failed to restrict SHAP explanation to top features "
                    "(sample=%s); plot will include all features. Error: %s",
                    display_features[:5],
                    exc,
                )
                # Keep using full feature list if trimming fails
                return explanation

        shap_shape = getattr(shap_values, "shape", None)
        class_labels = list(getattr(model, "classes_", []))
        shap_outputs = []
        if shap_shape is not None and len(shap_shape) == 3:
            output_count = shap_shape[2]
            LOG.info("Detected multi-output SHAP explanation with %s classes.", output_count)
            for class_idx in range(output_count):
                try:
                    class_expl = shap_values[..., class_idx]
                except Exception as exc:
                    LOG.warning(
                        "Failed to extract SHAP explanation for class index %s: %s",
                        class_idx,
                        exc,
                    )
                    continue
                label = (
                    class_labels[class_idx]
                    if class_labels and class_idx < len(class_labels)
                    else class_idx
                )
                shap_outputs.append((class_idx, label, class_expl))
        else:
            shap_outputs.append((None, None, shap_values))

        if not shap_outputs:
            LOG.error("No SHAP outputs available for plotting.")
            self.shap_model_name = None
            return

        # --- Plot SHAP summary (one per class if needed) ---
        for class_idx, class_label, class_expl in shap_outputs:
            expl_to_plot = _limit_explanation_features(class_expl)
            suffix = ""
            plot_key = "shap_summary"
            if class_idx is not None:
                safe_label = str(class_label).replace("/", "_").replace(" ", "_")
                suffix = f"_class_{safe_label}"
                plot_key = f"shap_summary_class_{safe_label}"
            out_filename = f"shap_summary{suffix}.png"
            out_path = os.path.join(self.output_dir, out_filename)
            plt.figure()
            shap.plots.beeswarm(expl_to_plot, max_display=max_display, show=False)
            title = f"SHAP Summary for {model.__class__.__name__}"
            if class_idx is not None:
                title += f" (class {class_label})"
            plt.title(f"{title} (top {max_display} features)")
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            self.plots[plot_key] = out_path

        # --- Log summary ---
        LOG.info(
            "SHAP summary completed with %s rows and %s features "
            "(displaying top %s) across %s output(s).",
            X_data.shape[0],
            X_data.shape[1],
            max_display,
            len(shap_outputs),
        )

    def generate_html_report(self):
        LOG.info("Generating HTML report")
        plots_html = ""
        for plot_name, plot_path in self.plots.items():
            if plot_name == "tree_importance" and not getattr(
                self, "tree_model_name", None
            ):
                continue
            encoded_image = self.encode_image_to_base64(plot_path)
            if plot_name == "tree_importance" and getattr(
                self, "tree_model_name", None
            ):
                section_title = f"Feature importance from {self.tree_model_name}"
            elif plot_name == "shap_summary":
                section_title = (
                    f"SHAP Summary from {getattr(self, 'shap_model_name', 'model')}"
                )
            elif plot_name.startswith("shap_summary_class_"):
                class_label = plot_name.replace("shap_summary_class_", "")
                section_title = (
                    f"SHAP Summary for class {class_label} "
                    f"({getattr(self, 'shap_model_name', 'model')})"
                )
            else:
                section_title = plot_name
            plots_html += f"""
            <div class="plot" id="{plot_name}" style="text-align:center;margin-bottom:24px;">
                <h2>{section_title}</h2>
                <img src="data:image/png;base64,{encoded_image}" alt="{plot_name}"
                     style="max-width:95%;height:auto;display:block;margin:0 auto;border:1px solid #ddd;padding:8px;background:#fff;">
            </div>
            """
        return f"{plots_html}"

    def encode_image_to_base64(self, img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def run(self):
        if (
            self.exp is None
            or not hasattr(self.exp, "is_setup")
            or not self.exp.is_setup
        ):
            self.setup_pycaret()
        self.save_tree_importance()
        self.save_shap_values()
        return self.generate_html_report()
