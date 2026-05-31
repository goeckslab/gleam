import base64
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
from explainability_limits import limit_explainability_data
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
        polynomial_features=False,
    ):
        self.task_type = task_type
        self.output_dir = output_dir
        self.exp = exp
        self.best_model = best_model
        self._skip_messages = []
        self.shap_total_features = None
        self.shap_used_features = None
        self.shap_total_rows = None
        self.shap_used_rows = None
        self.shap_scope = None
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
        self.polynomial_features = bool(polynomial_features)

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

        row_cap = max_samples
        if self.max_shap_rows is not None:
            row_cap = self.max_shap_rows if row_cap is None else min(row_cap, self.max_shap_rows)
        X_data, _, scope = limit_explainability_data(
            X_data,
            model=model,
            max_features=(
                self.max_plot_features
                if max_features is None
                else min(max_features, self.max_plot_features or max_features)
            ),
            max_rows=row_cap,
            random_seed=42,
            polynomial_features=self.polynomial_features,
            context="Custom SHAP",
        )
        self.shap_scope = scope
        self.shap_total_rows = scope.total_rows
        self.shap_used_rows = scope.used_rows
        self.shap_total_features = scope.total_features
        self.shap_used_features = scope.used_features
        display_features = list(X_data.columns)
        n_rows, n_features = X_data.shape

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
        predict_fn = self._get_predict_fn(model)

        # Optimized explainer based on model type
        explainer, explainer_label, tree_based = self._choose_shap_explainer(
            model, bg, predict_fn
        )
        if explainer is None:
            LOG.warning("No suitable SHAP explainer for model %s; skipping SHAP.", model)
            self.shap_model_name = None
            return

        try:
            shap_values = explainer(X_data)
            self.shap_model_name = explainer.__class__.__name__
        except Exception as e:
            error_message = str(e)
            needs_tree_fallback = (
                tree_based
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
                    agnostic_explainer = shap.Explainer(
                        predict_fn, bg, algorithm="permutation"
                    )
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

    def _get_predict_fn(self, model):
        if hasattr(model, "predict_proba"):
            return model.predict_proba
        if hasattr(model, "decision_function"):
            return model.decision_function
        return model.predict

    def _choose_shap_explainer(self, model, bg, predict_fn):
        """
        Select a SHAP explainer following the prescribed priority order for
        algorithms. Returns (explainer, label, is_tree_based).
        """
        if model is None:
            return None, None, False

        name = model.__class__.__name__
        lname = name.lower()
        task = getattr(self, "task_type", None)

        def _permutation(fn):
            return shap.Explainer(fn, bg, algorithm="permutation")

        if task == "classification":
            # 1) Logistic Regression
            if "logisticregression" in lname:
                return _permutation(model.predict_proba), "permutation-proba", False

            # 2) Ridge Classifier
            if "ridgeclassifier" in lname:
                fn = (
                    model.decision_function
                    if hasattr(model, "decision_function")
                    else predict_fn
                )
                return _permutation(fn), "permutation-decision_function", False

            # 3) LDA
            if "lineardiscriminantanalysis" in lname:
                return _permutation(model.predict_proba), "permutation-proba", False

            # 4) Random Forest
            if "randomforestclassifier" in lname:
                return (
                    shap.TreeExplainer(
                        model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 5) Gradient Boosting
            if "gradientboostingclassifier" in lname:
                return (
                    shap.TreeExplainer(
                        model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 6) AdaBoost
            if "adaboostclassifier" in lname:
                fn = model.predict_proba if hasattr(model, "predict_proba") else predict_fn
                return _permutation(fn), "permutation-adaboost", False

            # 7) Extra Trees
            if "extratreesclassifier" in lname:
                return (
                    shap.TreeExplainer(
                        model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 8) LightGBM
            if "lgbmclassifier" in lname:
                return (
                    shap.TreeExplainer(
                        model,
                        bg,
                        model_output="raw",
                        feature_perturbation="tree_path_dependent",
                        n_jobs=-1,
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 9) XGBoost
            if "xgbclassifier" in lname:
                return (
                    shap.TreeExplainer(
                        model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 10) CatBoost (classifier)
            if "catboost" in lname:
                return (
                    shap.TreeExplainer(
                        model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 11) KNN
            if "kneighborsclassifier" in lname:
                return _permutation(model.predict_proba), "permutation-proba", False

            # 12) SVM - linear kernel
            if "svc" in lname or "svm" in lname:
                kernel = getattr(model, "kernel", None)
                if kernel == "linear":
                    return shap.LinearExplainer(model, bg), "linear", False
                return _permutation(predict_fn), "permutation-svm", False

            # 13) Decision Tree
            if "decisiontreeclassifier" in lname:
                return (
                    shap.TreeExplainer(
                        model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                    ),
                    "tree_path_dependent",
                    True,
                )

            # 14) Naive Bayes
            if "naive_bayes" in lname or lname.endswith("nb"):
                fn = model.predict_proba if hasattr(model, "predict_proba") else predict_fn
                return _permutation(fn), "permutation-proba", False

            # 15) QDA
            if "quadraticdiscriminantanalysis" in lname:
                return _permutation(model.predict_proba), "permutation-proba", False

            # 16) Dummy
            if "dummyclassifier" in lname:
                return None, None, False

            # Default classification: permutation on predict_fn
            return _permutation(predict_fn), "permutation-default", False

        # Regression path
        # Linear family
        linear_keys = [
            "linearregression",
            "lasso",
            "ridge",
            "elasticnet",
            "lars",
            "lassolars",
            "orthogonalmatchingpursuit",
            "bayesianridge",
            "ardregression",
            "passiveaggressiveregressor",
            "theilsenregressor",
            "huberregressor",
        ]
        if any(k in lname for k in linear_keys):
            return shap.LinearExplainer(model, bg), "linear", False

        # Kernel ridge / SVR / KNN / MLP / RANSAC (model-agnostic)
        if "kernelridge" in lname:
            return _permutation(predict_fn), "permutation-kernelridge", False
        if "svr" in lname or "svm" in lname:
            kernel = getattr(model, "kernel", None)
            if kernel == "linear":
                return shap.LinearExplainer(model, bg), "linear", False
            return _permutation(predict_fn), "permutation-svr", False
        if "kneighborsregressor" in lname:
            return _permutation(predict_fn), "permutation-knn", False
        if "mlpregressor" in lname:
            return _permutation(predict_fn), "permutation-mlp", False
        if "ransacregressor" in lname:
            return _permutation(predict_fn), "permutation-ransac", False

        # Tree-based regressors
        tree_class_names = [
            "decisiontreeregressor",
            "randomforestregressor",
            "extratreesregressor",
            "gradientboostingregressor",
        ]
        if any(k in lname for k in tree_class_names):
            return (
                shap.TreeExplainer(
                    model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                ),
                "tree_path_dependent",
                True,
            )
        if "adaboostregressor" in lname:
            return _permutation(predict_fn), "permutation-adaboost", False

        # Boosting libraries
        if "lgbmregressor" in lname or "lightgbm" in lname:
            return (
                shap.TreeExplainer(
                    model,
                    bg,
                    model_output="raw",
                    feature_perturbation="tree_path_dependent",
                    n_jobs=-1,
                ),
                "tree_path_dependent",
                True,
            )
        if "xgbregressor" in lname or "xgboost" in lname:
            return (
                shap.TreeExplainer(
                    model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                ),
                "tree_path_dependent",
                True,
            )
        if "catboost" in lname:
            return (
                shap.TreeExplainer(
                    model, bg, feature_perturbation="tree_path_dependent", n_jobs=-1
                ),
                "tree_path_dependent",
                True,
            )

        # Default regression: model-agnostic permutation explainer
        return _permutation(predict_fn), "permutation-default", False

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
