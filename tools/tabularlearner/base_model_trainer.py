import base64
import json
import logging
import shutil
import tempfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from feature_help_modal import get_feature_metrics_help_modal
from feature_importance import FeatureImportanceAnalyzer
from sklearn.metrics import average_precision_score
from utils import (
    add_hr_to_html,
    add_plot_to_html,
    build_tabbed_html,
    encode_image_to_base64,
    get_html_closing,
    get_html_template,
)

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [
            {str(k): _json_safe(v) for k, v in row.items()}
            for row in value.replace({np.nan: None}).to_dict(orient="records")
        ]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _df_to_json_records(df: pd.DataFrame):
    if df is None:
        return []
    df_clean = df.replace({np.nan: None})
    records = df_clean.to_dict(orient="records")
    return [
        {str(k): _json_safe(v) for k, v in record.items()}
        for record in records
    ]


class BaseModelTrainer:
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
        self.exp = None
        self.input_file = input_file
        self.target_col = target_col
        self.output_dir = output_dir
        self.task_type = task_type
        self.random_seed = random_seed
        self.data = None
        self.target = None
        self.best_model = None
        self.results = None
        self.tuning_results = None
        self.features_name = None
        self.plots = {}
        self.explainer_plots = {}
        self.plots_explainer_html = None
        self.trees = []
        self.user_kwargs = kwargs.copy()
        for key, value in self.user_kwargs.items():
            setattr(self, key, value)
        self.setup_params = {}
        self.test_file = test_file
        self.test_data = None

        if not self.output_dir:
            raise ValueError(
                "output_dir must be specified and not None"
            )

        # Warn about irrelevant kwargs for the task type
        if self.task_type == "regression" and (
            "probability_threshold" in self.user_kwargs
        ):
            LOG.warning(
                "probability_threshold is ignored for regression tasks."
            )

        LOG.info(f"Model kwargs: {self.__dict__}")

    def load_data(self):
        LOG.info(f"Loading data from {self.input_file}")
        self.data = pd.read_csv(
            self.input_file, sep=None, engine="python"
        )
        self.data.columns = self.data.columns.str.replace(".", "_")

        names = self.data.columns.to_list()
        LOG.info(f"Original dataset columns: {names}")

        target_index = int(self.target_col) - 1
        num_cols = len(names)
        if target_index < 0 or target_index >= num_cols:
            raise ValueError(
                f"Target column number {self.target_col} is invalid. "
                f"Please select a number between 1 and {num_cols}."
            )

        self.target = names[target_index]

        # Conditional drop: only if 'prediction_label' exists and is not
        # the target
        if "prediction_label" in self.data.columns and (
            self.data.columns[target_index] != "prediction_label"
        ):
            LOG.info(
                "Dropping 'prediction_label' column as it's not the target."
            )
            self.data = self.data.drop(columns=["prediction_label"])
        else:
            if self.target == "prediction_label":
                LOG.warning(
                    "Using 'prediction_label' as target column. "
                    "This may not be intended if it's a previous prediction."
                )

        numeric_cols = self.data.select_dtypes(
            include=["number"]
        ).columns
        non_numeric_cols = self.data.select_dtypes(
            exclude=["number"]
        ).columns
        self.data[numeric_cols] = self.data[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        if len(non_numeric_cols) > 0:
            LOG.info(
                f"Non-numeric columns found: {non_numeric_cols.tolist()}"
            )

        # Update names after possible drop
        names = self.data.columns.to_list()
        LOG.info(f"Dataset columns after processing: {names}")

        self.features_name = [n for n in names if n != self.target]

        if getattr(self, "missing_value_strategy", None):
            strat = self.missing_value_strategy
            if strat == "mean":
                self.data = self.data.fillna(
                    self.data.mean(numeric_only=True)
                )
            elif strat == "median":
                self.data = self.data.fillna(
                    self.data.median(numeric_only=True)
                )
            elif strat == "drop":
                self.data = self.data.dropna()
        else:
            self.data = self.data.fillna(
                self.data.median(numeric_only=True)
            )

        if self.test_file:
            LOG.info(f"Loading test data from {self.test_file}")
            df_test = pd.read_csv(
                self.test_file, sep=None, engine="python"
            )
            df_test.columns = df_test.columns.str.replace(".", "_")
            self.test_data = df_test

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        self.setup_params = {
            "target": self.target,
            "session_id": self.random_seed,
            "html": True,
            "log_experiment": False,
            "system_log": False,
            "index": False,
        }
        if self.test_data is not None:
            self.setup_params["test_data"] = self.test_data
        for attr in [
            "train_size",
            "normalize",
            "feature_selection",
            "remove_outliers",
            "remove_multicollinearity",
            "polynomial_features",
            "feature_interaction",
            "feature_ratio",
            "fix_imbalance",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                self.setup_params[attr] = val
        if getattr(self, "cross_validation_folds", None) is not None:
            self.setup_params["fold"] = self.cross_validation_folds
        LOG.info(self.setup_params)

        if self.task_type == "classification":
            from pycaret.classification import ClassificationExperiment

            self.exp = ClassificationExperiment()
        elif self.task_type == "regression":
            from pycaret.regression import RegressionExperiment

            self.exp = RegressionExperiment()
        else:
            raise ValueError(
                "task_type must be 'classification' or 'regression'"
            )

        self.exp.setup(self.data, **self.setup_params)
        self.setup_params.update(self.user_kwargs)

    def train_model(self):
        LOG.info("Training and selecting the best model")
        if self.task_type == "classification":
            self.exp.add_metric(
                id="PR-AUC-Weighted",
                name="PR-AUC-Weighted",
                target="pred_proba",
                score_func=average_precision_score,
                average="weighted",
            )
        # Build arguments for compare_models()
        compare_kwargs = {}
        if getattr(self, "models", None):
            compare_kwargs["include"] = self.models

        # Respect explicit cross-validation flag
        if getattr(self, "cross_validation", None) is not None:
            compare_kwargs["cross_validation"] = self.cross_validation

        # Respect explicit fold count
        if getattr(self, "cross_validation_folds", None) is not None:
            compare_kwargs["fold"] = self.cross_validation_folds

        LOG.info(f"compare_models kwargs: {compare_kwargs}")
        self.best_model = self.exp.compare_models(**compare_kwargs)
        self.results = self.exp.pull()
        if getattr(self, "tune_model", False):
            LOG.info("Tuning hyperparameters of the best model")
            self.best_model = self.exp.tune_model(self.best_model)
            self.tuning_results = self.exp.pull()

        if self.task_type == "classification":
            self.results.rename(columns={"AUC": "ROC-AUC"}, inplace=True)

        prob_thresh = getattr(self, "probability_threshold", None)
        if self.task_type == "classification" and (
            prob_thresh is not None
        ):
            _ = self.exp.predict_model(
                self.best_model, probability_threshold=prob_thresh
            )
        else:
            _ = self.exp.predict_model(self.best_model)

        self.test_result_df = self.exp.pull()
        if self.task_type == "classification":
            self.test_result_df.rename(
                columns={"AUC": "ROC-AUC"}, inplace=True
            )

    def save_model(self):
        hdf5_path = Path(self.output_dir) / "pycaret_model.h5"
        with h5py.File(hdf5_path, "w") as f:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                joblib.dump(self.best_model, tmp.name)
                tmp.seek(0)
                model_bytes = tmp.read()
            f.create_dataset("model", data=np.void(model_bytes))

    def _store_plot(self, src_path: str, plot_key: str):
        if not src_path:
            return None

        src = Path(src_path)
        if not src.exists():
            LOG.warning(f"Plot path {src_path} does not exist; skipping copy.")
            return None

        ext = src.suffix or ".png"
        dest = Path(self.output_dir) / f"plot_{plot_key}{ext}"

        try:
            if src.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            stored_path = dest
        except Exception as exc:
            LOG.warning(
                f"Failed to copy plot {plot_key} from {src} to {dest}: {exc}. Using original path."
            )
            stored_path = src

        self.plots[plot_key] = str(stored_path)
        return self.plots[plot_key]

    def encode_image_to_base64(self, img_path: str) -> str:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _prepare_setup_summary(self, n_train, total_rows):
        all_params = self.setup_params.copy()
        if self.task_type == "classification" and (
            hasattr(self, "probability_threshold")
        ):
            all_params["probability_threshold"] = (
                self.probability_threshold
            )
        display_keys = [
            "Target",
            "Session ID",
            "Train Size",
            "Normalize",
            "Feature Selection",
            "Cross Validation",
            "Cross Validation Folds",
            "Remove Outliers",
            "Remove Multicollinearity",
            "Polynomial Features",
            "Fix Imbalance",
            "Models",
            "Probability Threshold",
        ]
        setup_rows = []
        for key in display_keys:
            pk = key.lower().replace(" ", "_")
            v = all_params.get(pk)
            if key == "Train Size":
                if v is not None:
                    frac = float(v)
                elif total_rows and n_train is not None:
                    frac = n_train / total_rows if total_rows else 0
                else:
                    frac = 0
                train_rows = int(n_train) if n_train is not None else 0
                dv = f"{frac:.2f} ({train_rows} rows)"
            elif key in {
                "Normalize",
                "Feature Selection",
                "Cross Validation",
                "Remove Outliers",
                "Remove Multicollinearity",
                "Polynomial Features",
                "Fix Imbalance",
            }:
                dv = bool(v)
            elif key == "Cross Validation Folds":
                dv = v if v is not None else "None"
            elif key == "Models":
                if isinstance(v, (list, tuple)):
                    dv = ", ".join(map(str, v))
                else:
                    dv = "None"
            elif key == "Probability Threshold":
                dv = f"{v:.2f}" if v is not None else "0.5"
            else:
                dv = v if v is not None else "None"
            setup_rows.append({"Parameter": key, "Value": dv})
        if hasattr(self.exp, "_fold_metric"):
            setup_rows.append(
                {"Parameter": "best_model_metric", "Value": self.exp._fold_metric}
            )
        df_setup = pd.DataFrame(setup_rows, columns=["Parameter", "Value"])
        return setup_rows, df_setup

    def save_html_report(self):
        LOG.info("Saving HTML report")

        # 1) Determine best model name
        try:
            best_model_name = str(self.results.iloc[0]["Model"])
        except Exception:
            best_model_name = type(self.best_model).__name__
        LOG.info(f"Best model determined as: {best_model_name}")

        # 2) Compute training sample count
        try:
            n_train = self.exp.X_train.shape[0]
        except Exception:
            n_train = getattr(
                self.exp, "X_train_transformed", pd.DataFrame()
            ).shape[0]
        total_rows = self.data.shape[0]

        # 3) Build setup parameters table
        _, df_setup = self._prepare_setup_summary(n_train, total_rows)
        df_setup.to_csv(
            Path(self.output_dir) / "setup_params.csv", index=False
        )

        # 4) Persist CSVs
        self.results.to_csv(
            Path(self.output_dir) / "comparison_results.csv",
            index=False
        )
        self.test_result_df.to_csv(
            Path(self.output_dir) / "test_results.csv", index=False
        )
        pd.DataFrame(
            self.best_model.get_params().items(),
            columns=["Parameter", "Value"]
        ).to_csv(Path(self.output_dir) / "best_model.csv", index=False)

        if self.tuning_results is not None:
            self.tuning_results.to_csv(
                Path(self.output_dir) / "tuning_results.csv",
                index=False
            )

        # 5) Header
        header = f"<h2>Best Model: {best_model_name}</h2>"

        # — Validation Summary & Configuration —
        val_df = self.results.copy()
        # mapping raw plot keys to user-friendly titles
        plot_title_map = {
            "learning": "Learning Curve",
            "vc": "Validation Curve",
            "calibration": "Calibration Curve",
            "dimension": "Dimensionality Reduction",
            "manifold": "Manifold Learning",
            "rfe": "Recursive Feature Elimination",
            "threshold": "Threshold Plot",
            "percentage_above_below": "Percentage Above vs. Below Cutoff",
            "class_report": "Classification Report",
            "pr_auc": "Precision-Recall AUC",
            "roc_auc": "Receiver Operating Characteristic AUC",
            "residuals": "Residuals Distribution",
            "error": "Prediction Error Distribution",
        }
        val_df.drop(
            columns=["TT (Ec)", "TT (Sec)"], errors="ignore", inplace=True
        )
        summary_html = (
            header
            + "<h2>Train & Validation Summary</h2>"
            + '<div class="table-wrapper">'
            + val_df.to_html(index=False, classes="table sortable")
            + "</div>"
        )

        if self.tuning_results is not None:
            tuning_df = self.tuning_results.copy()
            tuning_df.drop(
                columns=["TT (Sec)"], errors="ignore", inplace=True
            )
            summary_html += (
                f"<h2>{best_model_name}: Tuning Summary</h2>"
                + '<div class="table-wrapper">'
                + tuning_df.to_html(index=False, classes="table sortable")
                + "</div>"
            )

        summary_html += (
            "<h2>Setup Parameters</h2>"
            + '<div class="table-wrapper">'
            + df_setup.to_html(index=False, classes="table sortable")
            + "</div>"
            # — Hyperparameters
            + "<h2>Best Model Hyperparameters</h2>"
            + '<div class="table-wrapper">'
            + pd.DataFrame(
                self.best_model.get_params().items(),
                columns=["Parameter", "Value"]
            ).to_html(index=False, classes="table sortable")
            + "</div>"
        )

        # choose summary plots based on task type
        if self.task_type == "classification":
            summary_plots = [
                "learning",
                "vc",
                "calibration",
                "dimension",
                "manifold",
                "rfe",
                "threshold",
                "percentage_above_below",
            ]
        else:
            summary_plots = ["learning", "vc", "parameter", "residuals"]

        for name in summary_plots:
            if name in self.plots:
                summary_html += "<hr>"
                b64 = encode_image_to_base64(self.plots[name])
                title = plot_title_map.get(
                    name, name.replace("_", " ").title()
                )
                summary_html += (
                    '<div class="plot">'
                    f"<h2>{title}</h2>"
                    f'<img src="data:image/png;base64,{b64}" '
                    'style="max-width:90%;max-height:600px;'
                    'border:1px solid #ddd;"/>'
                    "</div>"
                )

        # — Test Summary —
        test_html = (
            header
            + '<div class="table-wrapper">'
            + self.test_result_df.to_html(
                index=False, classes="table sortable"
            )
            + "</div>"
        )
        if self.task_type == "regression":
            try:
                y_true = (
                    pd.Series(self.exp.y_test_transformed)
                    .reset_index(drop=True)
                    .rename("True")
                )
                y_pred = pd.Series(
                    self.best_model.predict(
                        self.exp.X_test_transformed
                    )
                ).rename("Predicted")
                df_tp = pd.concat([y_true, y_pred], axis=1)
                test_html += "<h2>True vs Predicted Values</h2>"
                test_html += (
                    '<div class="table-wrapper" '
                    'style="max-height:400px; overflow-y:auto;">'
                    + df_tp.head(50).to_html(
                        index=False, classes="table sortable"
                    )
                    + "</div>"
                    + add_hr_to_html()
                )
            except Exception as e:
                LOG.warning(
                    f"Could not generate True vs Predicted table: {e}"
                )

        # 5a) Explainer-substituted plots in order
        if self.task_type == "regression":
            test_order = ["residuals"]
        else:
            test_order = [
                "confusion_matrix",
                "roc_auc",
                "pr_auc",
                "lift_curve",
                "threshold",
                "cumulative_precision",
            ]
        for key in test_order:
            fig_or_fn = self.explainer_plots.pop(key, None)
            if fig_or_fn is not None:
                fig = fig_or_fn() if callable(fig_or_fn) else fig_or_fn
                title = plot_title_map.get(
                    key, key.replace("_", " ").title()
                )
                test_html += (
                    f"<h2>{title}</h2>" + add_plot_to_html(fig)
                    + add_hr_to_html()
                )
        # 5b) Remaining PyCaret test plots
        for name, path in self.plots.items():
            # classification: include only the small extras, before
            # skipping anything
            if self.task_type == "classification" and (
                name in {
                    "threshold",
                    "pr_auc",
                    "class_report",
                }
            ):
                title = plot_title_map.get(
                    name, name.replace("_", " ").title()
                )
                b64 = encode_image_to_base64(path)
                test_html += (
                    f"<h2>{title}</h2>"
                    "<div class='plot'>"
                    f"<img src='data:image/png;base64,{b64}' "
                    "style='max-width:90%;max-height:600px;"
                    "border:1px solid #ddd;'/>"
                    "</div>" + add_hr_to_html()
                )
                continue

            # regression: explicitly include the 'error' plot,
            # before skipping
            if self.task_type == "regression" and (
                name == "error"
            ):
                title = plot_title_map.get(
                    "error", "Prediction Error Distribution"
                )
                b64 = encode_image_to_base64(path)
                test_html += (
                    f"<h2>{title}</h2>"
                    "<div class='plot'>"
                    f"<img src='data:image/png;base64,{b64}' "
                    "style='max-width:90%;max-height:600px;"
                    "border:1px solid #ddd;'/>"
                    "</div>" + add_hr_to_html()
                )
                continue

            # now skip any plots already rendered via test_order
            if name in test_order:
                continue

        # — Feature Importance —
        feature_html = header

        # 6a) PyCaret’s default feature importances
        feature_html += FeatureImportanceAnalyzer(
            data=self.data,
            target_col=self.target_col,
            task_type=self.task_type,
            output_dir=self.output_dir,
            exp=self.exp,
            best_model=self.best_model,
        ).run()

        # 6b) Explainer SHAP importances
        for key in ["shap_mean", "shap_perm"]:
            fig_or_fn = self.explainer_plots.pop(key, None)
            if fig_or_fn is not None:
                fig = fig_or_fn() if callable(fig_or_fn) else fig_or_fn
                # give SHAP plots explicit titles
                title = (
                    "Mean Absolute SHAP Value Impact"
                    if key == "shap_mean"
                    else "Permutation Feature Importance"
                )
                feature_html += (
                    f"<h2>{title}</h2>" + add_plot_to_html(fig)
                    + add_hr_to_html()
                )

        # 6c) PDPs last
        pdp_keys = sorted(
            k for k in self.explainer_plots if k.startswith("pdp__")
        )
        for k in pdp_keys:
            fig_or_fn = self.explainer_plots[k]
            fig = fig_or_fn() if callable(fig_or_fn) else fig_or_fn
            # extract feature name
            feature = k.split("__", 1)[1]
            title = f"Partial Dependence for {feature}"
            feature_html += (
                f"<h2>{title}</h2>" + add_plot_to_html(fig)
                + add_hr_to_html()
            )
        # 7) Assemble final HTML (three tabs)
        html = get_html_template()
        html += "<h1>Tabular Learner Model Report</h1>"
        html += build_tabbed_html(summary_html, test_html, feature_html)
        html += get_feature_metrics_help_modal()
        html += get_html_closing()

        # 8) Write out
        (Path(self.output_dir) / "comparison_result.html").write_text(
            html, encoding="utf-8"
        )
        LOG.info(
            f"HTML report generated at: "
            f"{self.output_dir}/comparison_result.html"
        )

    def save_metrics_json(self):
        LOG.info("Saving metrics JSON report")

        try:
            best_model_name = str(self.results.iloc[0]["Model"])
        except Exception:
            best_model_name = type(self.best_model).__name__

        try:
            n_train = self.exp.X_train.shape[0]
        except Exception:
            n_train = getattr(
                self.exp, "X_train_transformed", pd.DataFrame()
            ).shape[0]
        total_rows = self.data.shape[0] if self.data is not None else None

        setup_rows, _ = self._prepare_setup_summary(n_train, total_rows)

        best_model_params = {
            str(k): _json_safe(v)
            for k, v in self.best_model.get_params().items()
        }

        metrics_block = {
            "train_validation": _df_to_json_records(self.results),
            "test": _df_to_json_records(self.test_result_df),
        }
        if self.tuning_results is not None:
            metrics_block["tuning"] = _df_to_json_records(self.tuning_results)

        payload = {
            "task_type": self.task_type,
            "target_column": self.target,
            "random_seed": self.random_seed,
            "data_summary": {
                "total_rows": int(total_rows)
                if total_rows is not None
                else None,
                "train_rows": int(n_train)
                if n_train is not None
                else None,
                "test_dataset_provided": bool(self.test_data is not None),
            },
            "best_model": {
                "name": best_model_name,
                "metric": getattr(self.exp, "_fold_metric", None),
                "hyperparameters": best_model_params,
            },
            "setup_parameters": [
                {
                    "Parameter": row["Parameter"],
                    "Value": _json_safe(row["Value"]),
                }
                for row in setup_rows
            ],
            "metrics": metrics_block,
        }

        prob_thresh = getattr(self, "probability_threshold", None)
        if self.task_type == "classification" and prob_thresh is not None:
            payload["best_model"]["probability_threshold"] = prob_thresh

        metrics_path = Path(self.output_dir) / "metrics.json"
        metrics_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        LOG.info(f"Metrics JSON generated at: {metrics_path}")

    def save_dashboard(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_plots_explainer(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_tree_plots(self):
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor
        )
        from xgboost import XGBClassifier, XGBRegressor
        from explainerdashboard.explainers import RandomForestExplainer

        LOG.info("Generating tree plots")
        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed

        if isinstance(
            self.best_model, (RandomForestClassifier, RandomForestRegressor)
        ):
            n_trees = self.best_model.n_estimators
        elif isinstance(self.best_model, (XGBClassifier, XGBRegressor)):
            n_trees = len(self.best_model.get_booster().get_dump())
        else:
            LOG.warning("Tree plots not supported for this model type.")
            return

        explainer = RandomForestExplainer(self.best_model, X_test, y_test)
        for i in range(n_trees):
            fig = explainer.decisiontree_encoded(tree_idx=i, index=0)
            self.trees.append(fig)

    def run(self):
        self.load_data()
        self.setup_pycaret()
        self.train_model()
        self.save_model()
        self.generate_plots()
        self.generate_plots_explainer()
        self.generate_tree_plots()
        self.save_html_report()
        self.save_metrics_json()
        # self.save_dashboard()
