import base64
import logging
import tempfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from feature_help_modal import get_feature_metrics_help_modal
from feature_importance import FeatureImportanceAnalyzer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
        self.plot_feature_names = None
        self.plots = {}
        self.explainer_plots = {}
        self.plots_explainer_html = None
        self.trees = []
        self.user_kwargs = kwargs.copy()
        for key, value in self.user_kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "plot_feature_limit"):
            self.plot_feature_limit = 30
        self._shap_row_cap = None
        if getattr(self, "polynomial_features", False):
            # Keep feature importance responsive by trimming plots/SHAP rows
            try:
                limit_val = int(self.plot_feature_limit)
            except (TypeError, ValueError):
                limit_val = 30
            self.plot_feature_limit = min(limit_val, 15)
            self._shap_row_cap = 200
            LOG.info(
                "Polynomial features enabled; limiting feature plots to %s and SHAP rows to %s",
                self.plot_feature_limit,
                self._shap_row_cap,
            )
        self.imputed_training_data = None
        self._best_model_metric_used = None
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
        sample_id_column = getattr(self, "sample_id_column", None)
        if sample_id_column:
            if str(sample_id_column).isdigit():
                idx = int(sample_id_column) - 1
                if 0 <= idx < len(names):
                    resolved = names[idx]
                    if sample_id_column in names:
                        LOG.warning(
                            "Sample ID column value '%s' matches a header, but Galaxy data_column "
                            "inputs are interpreted as 1-based indices; using column #%s header '%s'.",
                            sample_id_column,
                            idx + 1,
                            resolved,
                        )
                    LOG.info(
                        "Sample ID column '%s' not found; using column #%s header '%s' instead.",
                        sample_id_column,
                        idx + 1,
                        resolved,
                    )
                    sample_id_column = resolved
                else:
                    raise ValueError(
                        f"Sample ID column index {sample_id_column} is invalid. "
                        f"Please select a number between 1 and {len(names)}."
                    )
            sample_id_column = sample_id_column.replace(".", "_")
            self.sample_id_column = sample_id_column
        else:
            self.sample_id_column = None
        self.sample_id_series = None

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

        sample_id_valid = False
        if sample_id_column:
            if sample_id_column not in self.data.columns:
                LOG.warning(
                    "Sample ID column '%s' not found; proceeding without group-aware split.",
                    sample_id_column,
                )
                sample_id_column = None
                self.sample_id_column = None
            elif sample_id_column == self.target:
                LOG.warning(
                    "Sample ID column '%s' matches target column; skipping group-aware split.",
                    sample_id_column,
                )
                sample_id_column = None
                self.sample_id_column = None
            else:
                sample_id_valid = True

        if self.test_file:
            LOG.info(f"Loading test data from {self.test_file}")
            df_test = pd.read_csv(
                self.test_file, sep=None, engine="python"
            )
            df_test.columns = df_test.columns.str.replace(".", "_")
            self.test_data = df_test

        if sample_id_valid and self.test_data is None:
            train_size = getattr(self, "train_size", None)
            if train_size is None:
                train_size = 0.7
            if train_size <= 0 or train_size >= 1:
                LOG.warning(
                    "Invalid train_size=%s; skipping group-aware split.",
                    train_size,
                )
            else:
                rng = np.random.RandomState(self.random_seed)

                def _allocate_split_counts(n_total: int, probs: list) -> list:
                    if n_total <= 0:
                        return [0 for _ in probs]
                    counts = [0 for _ in probs]
                    active = [i for i, p in enumerate(probs) if p > 0]
                    remainder = n_total
                    if active and n_total >= len(active):
                        for i in active:
                            counts[i] = 1
                        remainder -= len(active)
                    if remainder > 0:
                        probs_arr = np.array(probs, dtype=float)
                        probs_arr = probs_arr / probs_arr.sum()
                        raw = remainder * probs_arr
                        floors = np.floor(raw).astype(int)
                        for i, value in enumerate(floors.tolist()):
                            counts[i] += value
                        leftover = remainder - int(floors.sum())
                        if leftover > 0 and active:
                            frac = raw - floors
                            order = sorted(active, key=lambda i: (-frac[i], i))
                            for i in range(leftover):
                                counts[order[i % len(order)]] += 1
                    return counts

                def _choose_split(counts: list, targets: list, active: list) -> int:
                    remaining = [targets[i] - counts[i] for i in range(len(targets))]
                    best = max(active, key=lambda i: (remaining[i], -counts[i], -targets[i]))
                    if remaining[best] <= 0:
                        best = min(active, key=lambda i: counts[i])
                    return best

                probs = [train_size, 1.0 - train_size]
                targets = _allocate_split_counts(len(self.data), probs)
                counts = [0, 0]
                active = [0, 1]
                train_idx = []
                test_idx = []

                group_series = self.data[sample_id_column].astype(object)
                missing_mask = group_series.isna()
                if missing_mask.any():
                    group_series = group_series.copy()
                    group_series.loc[missing_mask] = [
                        f"__missing__{idx}" for idx in group_series.index[missing_mask]
                    ]
                group_to_indices = {}
                for idx, group_id in group_series.items():
                    group_to_indices.setdefault(group_id, []).append(idx)

                group_ids = sorted(group_to_indices.keys(), key=lambda x: str(x))
                rng.shuffle(group_ids)

                for group_id in group_ids:
                    split_idx = _choose_split(counts, targets, active)
                    counts[split_idx] += len(group_to_indices[group_id])
                    if split_idx == 0:
                        train_idx.extend(group_to_indices[group_id])
                    else:
                        test_idx.extend(group_to_indices[group_id])

                missing_splits = []
                if not train_idx:
                    missing_splits.append("train")
                if not test_idx:
                    missing_splits.append("test")
                if missing_splits:
                    LOG.warning(
                        "Group-aware split using '%s' produced empty %s set; "
                        "falling back to default split.",
                        sample_id_column,
                        " and ".join(missing_splits),
                    )
                else:
                    self.test_data = self.data.loc[test_idx].reset_index(drop=True)
                    self.data = self.data.loc[train_idx].reset_index(drop=True)
                    LOG.info(
                        "Applied group-aware split using '%s' (train=%s, test=%s).",
                        sample_id_column,
                        len(train_idx),
                        len(test_idx),
                    )

        if sample_id_valid:
            self.sample_id_series = self.data[sample_id_column].copy()
            if sample_id_column in self.data.columns:
                self.data = self.data.drop(columns=[sample_id_column])
            if self.test_data is not None and sample_id_column in self.test_data.columns:
                self.test_data = self.test_data.drop(columns=[sample_id_column])

        # Refresh feature lists after any sample-id column removal.
        names = self.data.columns.to_list()
        self.features_name = [n for n in names if n != self.target]
        self.plot_feature_names = self._select_plot_features(self.features_name)

    def _select_plot_features(self, all_features):
        limit = getattr(self, "plot_feature_limit", 30)
        if not isinstance(limit, int) or limit <= 0:
            LOG.info(
                "Feature plotting limit disabled (plot_feature_limit=%s).", limit
            )
            return all_features
        if len(all_features) <= limit:
            LOG.info(
                "Feature plotting limit not needed (%s features <= limit %s).",
                len(all_features),
                limit,
            )
            return all_features
        df = self.data[all_features].copy()
        numeric_cols = df.select_dtypes(include=["number"]).columns
        ranked = []
        if len(numeric_cols) > 0:
            variances = (
                df[numeric_cols]
                .var()
                .fillna(0)
                .abs()
                .sort_values(ascending=False)
            )
            ranked = variances.index.tolist()
        selected = []
        for col in ranked:
            if len(selected) >= limit:
                break
            selected.append(col)
        if len(selected) < limit:
            for col in all_features:
                if col in selected:
                    continue
                selected.append(col)
                if len(selected) >= limit:
                    break
        LOG.info(
            "Limiting feature-level plots to %s of %s available features (limit=%s).",
            len(selected),
            len(all_features),
            limit,
        )
        return selected

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
            "n_jobs",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                self.setup_params[attr] = val
        if getattr(self, "cross_validation_folds", None) is not None:
            self.setup_params["fold"] = self.cross_validation_folds
        LOG.info(self.setup_params)

        group_series = getattr(self, "sample_id_series", None)
        if group_series is not None and getattr(self, "cross_validation", None) is not False:
            n_groups = pd.Series(group_series).nunique(dropna=False)
            fold_count = getattr(self, "cross_validation_folds", None)
            if fold_count is not None and fold_count > n_groups:
                LOG.warning(
                    "cross_validation_folds=%s exceeds unique groups=%s; "
                    "skipping group-aware CV.",
                    fold_count,
                    n_groups,
                )
            else:
                self.setup_params["fold_strategy"] = "groupkfold"
                self.setup_params["fold_groups"] = pd.Series(group_series).reset_index(drop=True)
                LOG.info(
                    "Enabled group-aware CV with %s unique groups.",
                    n_groups,
                )

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
        self._capture_imputed_training_data()
        self.setup_params.update(self.user_kwargs)

    def _capture_imputed_training_data(self):
        """
        Cache the dataset as transformed/imputed by PyCaret so downstream
        components (e.g., feature importance) can operate on the exact data
        used for training.
        """
        if self.exp is None:
            return
        try:
            X_processed = self.exp.get_config("X_transformed").copy()
            y_processed = self.exp.get_config("y")
            if isinstance(y_processed, pd.Series):
                y_series = y_processed.reset_index(drop=True)
            else:
                y_series = pd.Series(y_processed)
            y_series.name = self.target
            X_processed = X_processed.reset_index(drop=True)
            self.imputed_training_data = pd.concat(
                [X_processed, y_series], axis=1
            )
            LOG.info(
                "Captured imputed training dataset from PyCaret "
                "(%s rows, %s features).",
                self.imputed_training_data.shape[0],
                self.imputed_training_data.shape[1] - 1,
            )
        except Exception as exc:
            LOG.warning(
                "Unable to capture processed training data from PyCaret: %s",
                exc,
            )
            self.imputed_training_data = None

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

        best_metric = getattr(self, "best_model_metric", None)
        if best_metric:
            compare_kwargs["sort"] = best_metric
            self._best_model_metric_used = best_metric
            LOG.info(f"Ranking models using metric: {best_metric}")

        LOG.info(f"compare_models kwargs: {compare_kwargs}")
        self.best_model = self.exp.compare_models(**compare_kwargs)
        if self._best_model_metric_used is None:
            self._best_model_metric_used = getattr(self.exp, "_fold_metric", None)
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

    def generate_plots(self):
        LOG.info("Generating PyCaret diagnostic pltos")

        # choose the right plots based on task type
        if self.task_type == "classification":
            plot_names = [
                "learning",
                "vc",
                "calibration",
                "dimension",
                "manifold",
                "rfe",
                "threshold",
                "percentage_above_below",
                "pr_auc",
                "roc_auc",
            ]
        else:
            plot_names = ["residuals", "vc", "parameter", "error",
                          "learning"]
        for name in plot_names:
            try:
                ax = self.exp.plot_model(
                    self.best_model, plot=name, save=False
                )
                out_path = Path(self.output_dir) / f"plot_{name}.png"
                fig = ax.get_figure()
                fig.savefig(out_path, bbox_inches="tight")
                self.plots[name] = str(out_path)
            except Exception as e:
                LOG.warning(f"Could not generate {name} plot: {e}")

    def encode_image_to_base64(self, img_path: str) -> str:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _decode_class_labels_for_display(self, values):
        """Map PyCaret-encoded class labels back to the original target values."""
        if self.task_type != "classification" or self.exp is None:
            return values

        try:
            from pycaret.utils.generic import get_label_encoder

            label_encoder = get_label_encoder(self.exp.pipeline)
        except Exception as exc:
            LOG.debug("Could not load label encoder for display labels: %s", exc)
            label_encoder = None

        if label_encoder is None:
            return values

        def _decode_array(arr):
            arr = np.asarray(arr, dtype=object)
            flat = arr.reshape(-1)
            decoded = flat.copy()
            mask = pd.notna(flat)
            if not mask.any():
                return arr

            encoded_vals = flat[mask]
            try:
                decoded_vals = label_encoder.inverse_transform(encoded_vals)
            except Exception:
                try:
                    numeric_vals = pd.to_numeric(encoded_vals, errors="raise")
                    encoded_vals = numeric_vals.astype(int)
                    decoded_vals = label_encoder.inverse_transform(encoded_vals)
                except Exception as exc:
                    LOG.debug(
                        "Could not inverse-transform class labels for display: %s",
                        exc,
                    )
                    return arr

            decoded[mask] = decoded_vals
            return decoded.reshape(arr.shape)

        if isinstance(values, pd.Series):
            decoded = _decode_array(values.to_numpy())
            return pd.Series(decoded, index=values.index, name=values.name)

        decoded = _decode_array(values)
        if isinstance(values, list):
            return decoded.tolist()
        if np.isscalar(values):
            return decoded.reshape(-1)[0]
        return decoded

    def _build_dataset_overview(self):
        """
        Build an HTML table showing label counts with labels as rows and splits
        (Train / Validation / Test) as columns. Each cell shows count and
        percentage of that split. Returns empty string for regression or when
        no label data is available.
        """
        if self.task_type != "classification":
            return ""

        def _safe_series(obj):
            try:
                return pd.Series(obj).reset_index(drop=True)
            except Exception:
                return None

        def _get_from_config(keys):
            if self.exp is None:
                return None
            for key in keys:
                try:
                    val = self.exp.get_config(key)
                except Exception:
                    val = getattr(self.exp, key, None)
                if val is not None:
                    return val
            return None

        # Prefer original-label PyCaret splits; fall back to transformed labels
        # only when originals are unavailable.
        X_train = _get_from_config(["X_train_transformed", "X_train"])
        y_train = _get_from_config(["y_train", "y_train_transformed"])
        y_test_cfg = _get_from_config(["y_test", "y_test_transformed"])

        if y_train is None and self.data is not None and self.target in self.data.columns:
            y_train = self.data[self.target]

        y_train_series = _safe_series(y_train)

        # Build a cross-validation generator to derive a validation subset size.
        cv_gen = self._get_cv_generator(y_train_series)
        y_train_fold = y_train_series
        y_val_fold = None
        if cv_gen is not None and y_train_series is not None:
            try:
                # Use the first fold to approximate Train/Validation split sizes.
                splitter = cv_gen.split(
                    pd.DataFrame(X_train).reset_index(drop=True)
                    if X_train is not None
                    else y_train_series,
                    y_train_series,
                )
                train_idx, val_idx = next(iter(splitter))
                y_train_fold = y_train_series.iloc[train_idx].reset_index(drop=True)
                y_val_fold = y_train_series.iloc[val_idx].reset_index(drop=True)
            except Exception as exc:
                LOG.warning("Could not derive validation split for dataset overview: %s", exc)

        # Test labels: prefer external/raw labels, then PyCaret original labels,
        # then transformed labels.
        if self.test_data is not None:
            if self.target in self.test_data.columns:
                y_test = self.test_data[self.target]
            elif y_test_cfg is not None:
                y_test = y_test_cfg
            else:
                y_test = None
        else:
            y_test = y_test_cfg

        split_map = {
            "Train": _safe_series(y_train_fold),
            "Validation": _safe_series(y_val_fold),
            "Test": _safe_series(y_test),
        }
        split_map = {
            name: (
                self._decode_class_labels_for_display(series)
                if series is not None
                else None
            )
            for name, series in split_map.items()
        }
        available = {k: v for k, v in split_map.items() if v is not None and not v.empty}
        if not available:
            return ""

        # Collect all labels across available splits (including NaN)
        label_pool = pd.concat(
            available.values(), ignore_index=True
        )
        labels = pd.unique(label_pool)

        def _count_for_label(series, label):
            if series is None or series.empty:
                return None, None
            total = len(series)
            if pd.isna(label):
                cnt = series.isna().sum()
            else:
                cnt = (series == label).sum()
            return int(cnt), total

        rows = []
        for label in labels:
            row = ["NaN" if pd.isna(label) else str(label)]
            for split_name in ["Train", "Validation", "Test"]:
                cnt, total = _count_for_label(split_map.get(split_name), label)
                if cnt is None or total is None:
                    cell = "—"
                else:
                    pct = (cnt / total * 100) if total else 0
                    cell = f"{cnt} ({pct:.1f}%)"
                row.append(cell)
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Label", "Train", "Validation", "Test"])
        df.sort_values("Label", inplace=True)

        return (
            "<h2>Dataset Overview</h2>"
            + '<div class="table-wrapper">'
            + df.to_html(
                index=False,
                classes=["table", "sortable", "table-dataset-overview"],
            )
            + "</div>"
        )

    def _predict_with_thresholds(self, X, y_true):
        """
        Generate predictions/probabilities for a split, respecting an optional
        probability threshold for binary tasks. Returns a dict with y_true,
        y_pred, y_scores (positive-class probs when available), pos_label,
        and neg_label.
        """
        if X is None or y_true is None:
            return None

        y_true_series = pd.Series(y_true).reset_index(drop=True)
        classes = list(getattr(self.best_model, "classes_", []))
        if not classes:
            try:
                classes = pd.unique(y_true_series).tolist()
            except Exception:
                classes = []
        if len(classes) > 1:
            try:
                pos_idx = classes.index(1)
            except Exception:
                pos_idx = 1
        else:
            pos_idx = 0
        pos_idx = min(pos_idx, len(classes) - 1) if classes else 0
        pos_label = (
            classes[pos_idx]
            if len(classes) > pos_idx and pos_idx >= 0
            else (classes[-1] if classes else 1)
        )
        neg_label = None
        if len(classes) >= 2:
            neg_candidates = [c for c in classes if c != pos_label]
            if neg_candidates:
                neg_label = neg_candidates[0]

        prob_thresh = getattr(self, "probability_threshold", None)
        y_scores = None
        try:
            proba = self.best_model.predict_proba(X)
            y_scores = np.asarray(proba) if proba is not None else None
        except Exception:
            y_scores = None

        try:
            if (
                prob_thresh is not None
                and not getattr(self.exp, "is_multiclass", False)
                and y_scores is not None
                and y_scores.ndim == 2
                and y_scores.shape[1] > 1
            ):
                pos_idx = min(pos_idx, y_scores.shape[1] - 1)
                neg_idx = 1 - pos_idx if y_scores.shape[1] > 1 else 0
                if neg_label is None and len(classes) > neg_idx:
                    neg_label = classes[neg_idx]
                y_pred = np.where(
                    y_scores[:, pos_idx] >= prob_thresh,
                    pos_label,
                    neg_label if neg_label is not None else 0,
                )
                y_scores = y_scores[:, pos_idx]
            else:
                y_pred = self.best_model.predict(X)
                if (
                    not getattr(self.exp, "is_multiclass", False)
                    and y_scores is not None
                    and y_scores.ndim == 2
                    and y_scores.shape[1] > 1
                ):
                    pos_idx = min(pos_idx, y_scores.shape[1] - 1)
                    y_scores = y_scores[:, pos_idx]
        except Exception as exc:
            LOG.warning(
                "Falling back to raw predict while computing performance summary: %s",
                exc,
            )
            try:
                y_pred = self.best_model.predict(X)
            except Exception as exc_inner:
                LOG.warning(
                    "Unable to score split after fallback prediction: %s",
                    exc_inner,
                )
                return None
            y_scores = None

        y_pred_series = pd.Series(y_pred).reset_index(drop=True)
        if y_scores is not None:
            y_scores = np.asarray(y_scores)
            if y_scores.ndim > 1 and y_scores.shape[1] == 1:
                y_scores = y_scores.ravel()
            if getattr(self.exp, "is_multiclass", False) and y_scores.ndim > 1:
                # Avoid passing multiclass score matrices to ROC/PR utilities
                y_scores = None

        return {
            "y_true": y_true_series,
            "y_pred": y_pred_series,
            "y_scores": y_scores,
            "pos_label": pos_label,
            "neg_label": neg_label,
        }

    def _get_cv_generator(self, y_series):
        """
        Build a cross-validation splitter that mirrors the experiment's
        configuration. Returns None when CV is disabled or not applicable.
        """
        if self.task_type != "classification":
            return None

        if getattr(self, "cross_validation", None) is False:
            return None

        try:
            cfg_gen = self.exp.get_config("fold_generator")
            if cfg_gen is not None:
                return cfg_gen
        except Exception:
            cfg_gen = None

        folds = (
            getattr(self, "cross_validation_folds", None)
            or self.setup_params.get("fold")
            or getattr(self.exp, "fold", None)
            or 10
        )
        try:
            folds = int(folds)
        except Exception:
            folds = 10

        try:
            y_series = pd.Series(y_series).reset_index(drop=True)
        except Exception:
            y_series = None
        if y_series is None or y_series.empty:
            return None

        if folds < 2:
            return None
        if len(y_series) < folds:
            folds = len(y_series)
        if folds < 2:
            return None

        try:
            from sklearn.model_selection import KFold, StratifiedKFold

            if self.task_type == "classification":
                return StratifiedKFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=self.random_seed,
                )
            return KFold(
                n_splits=folds,
                shuffle=True,
                random_state=self.random_seed,
            )
        except Exception as exc:
            LOG.warning("Could not build CV generator: %s", exc)
            return None

    def _get_cross_validated_predictions(self, X, y):
        """
        Generate cross-validated predictions for the validation split so we
        can report validation metrics for the selected best model.
        """
        if self.task_type != "classification":
            return None
        if getattr(self, "cross_validation", None) is False:
            return None
        if X is None or y is None:
            return None

        try:
            from sklearn.model_selection import cross_val_predict
        except Exception as exc:
            LOG.warning("cross_val_predict unavailable: %s", exc)
            return None

        y_series = pd.Series(y).reset_index(drop=True)
        if y_series.empty:
            return None

        cv_gen = self._get_cv_generator(y_series)
        if cv_gen is None:
            return None

        X_df = pd.DataFrame(X).reset_index(drop=True)
        if len(X_df) != len(y_series):
            X_df = X_df.iloc[: len(y_series)].reset_index(drop=True)

        classes = list(getattr(self.best_model, "classes_", []))
        if len(classes) > 1:
            try:
                pos_idx = classes.index(1)
            except Exception:
                pos_idx = 1
        else:
            pos_idx = 0
        pos_idx = min(pos_idx, len(classes) - 1) if classes else 0
        pos_label = (
            classes[pos_idx] if len(classes) > pos_idx else 1
        )
        neg_label = None
        if len(classes) >= 2:
            neg_candidates = [c for c in classes if c != pos_label]
            if neg_candidates:
                neg_label = neg_candidates[0]

        prob_thresh = getattr(self, "probability_threshold", None)
        n_jobs = getattr(self, "n_jobs", None)

        y_scores = None
        if not getattr(self.exp, "is_multiclass", False):
            try:
                proba = cross_val_predict(
                    self.best_model,
                    X_df,
                    y_series,
                    cv=cv_gen,
                    method="predict_proba",
                    n_jobs=n_jobs,
                )
                y_scores = np.asarray(proba)
            except Exception as exc:
                LOG.debug("Could not compute CV probabilities: %s", exc)

        y_pred = None
        if (
            prob_thresh is not None
            and not getattr(self.exp, "is_multiclass", False)
            and y_scores is not None
            and y_scores.ndim == 2
            and y_scores.shape[1] > 1
        ):
            pos_idx = min(pos_idx, y_scores.shape[1] - 1)
            neg_idx = 1 - pos_idx if y_scores.shape[1] > 1 else 0
            if neg_label is None and len(classes) > neg_idx:
                neg_label = classes[neg_idx]
            y_pred = np.where(
                y_scores[:, pos_idx] >= prob_thresh,
                pos_label,
                neg_label if neg_label is not None else 0,
            )
            y_scores = y_scores[:, pos_idx]
        else:
            try:
                y_pred = cross_val_predict(
                    self.best_model,
                    X_df,
                    y_series,
                    cv=cv_gen,
                    method="predict",
                    n_jobs=n_jobs,
                )
            except Exception as exc:
                LOG.warning(
                    "Could not compute cross-validated predictions: %s",
                    exc,
                )
                return None
            if (
                not getattr(self.exp, "is_multiclass", False)
                and y_scores is not None
                and y_scores.ndim == 2
                and y_scores.shape[1] > 1
            ):
                pos_idx = min(pos_idx, y_scores.shape[1] - 1)
                y_scores = y_scores[:, pos_idx]

        if y_scores is not None and getattr(self.exp, "is_multiclass", False):
            y_scores = None

        return {
            "y_true": y_series,
            "y_pred": pd.Series(y_pred).reset_index(drop=True),
            "y_scores": y_scores,
            "pos_label": pos_label,
            "neg_label": neg_label,
        }

    def _get_split_predictions_for_report(self):
        """
        Collect predictions/probabilities for Train/Validation/Test splits so the
        performance table can show consistent metrics across splits.
        """
        if self.task_type != "classification":
            return {}

        def _get_from_config(keys):
            for key in keys:
                try:
                    val = self.exp.get_config(key)
                except Exception:
                    val = getattr(self.exp, key, None)
                if val is not None:
                    return val
            return None

        X_train = _get_from_config(["X_train_transformed", "X_train"])
        y_train = _get_from_config(["y_train_transformed", "y_train"])
        X_holdout = _get_from_config(["X_test_transformed", "X_test"])
        y_holdout = _get_from_config(["y_test_transformed", "y_test"])

        predictions = {}

        # Train metrics (best model on training data)
        if X_train is not None and y_train is not None:
            try:
                train_preds = self._predict_with_thresholds(X_train, y_train)
                if train_preds is not None:
                    predictions["Train"] = train_preds
            except Exception as exc:
                LOG.warning(
                    "Could not score Train split for performance summary: %s",
                    exc,
                )

        # Validation metrics via cross-validation on training data
        try:
            val_preds = self._get_cross_validated_predictions(X_train, y_train)
            if val_preds is not None:
                predictions["Validation"] = val_preds
        except Exception as exc:
            LOG.warning(
                "Could not score Validation split for performance summary: %s",
                exc,
            )

        # Test metrics (holdout from single file, or provided test file)
        X_test = X_holdout
        y_test = y_holdout
        if (X_test is None or y_test is None) and self.test_data is not None:
            try:
                X_test = self.test_data.drop(columns=[self.target])
                y_test = self.test_data[self.target]
            except Exception as exc:
                LOG.warning(
                    "Could not prepare external test data for performance summary: %s",
                    exc,
                )

        if X_test is not None and y_test is not None:
            try:
                test_preds = self._predict_with_thresholds(X_test, y_test)
                if test_preds is not None:
                    predictions["Test"] = test_preds
            except Exception as exc:
                LOG.warning(
                    "Could not score Test split for performance summary: %s",
                    exc,
                )
        return predictions

    def _compute_metric_value(self, metric_name, preds, split_name):
        """
        Compute a single metric for a given split prediction bundle.
        """
        if preds is None:
            return None

        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        y_scores = preds.get("y_scores")
        pos_label = preds.get("pos_label")
        neg_label = preds.get("neg_label")
        is_multiclass = getattr(self.exp, "is_multiclass", False)

        def _format_binary_labels(series):
            if pos_label is None:
                return series
            try:
                return (series == pos_label).astype(int)
            except Exception:
                return series

        try:
            if metric_name == "Accuracy":
                return accuracy_score(y_true, y_pred)
            if metric_name == "ROC-AUC":
                if y_scores is None:
                    return None
                y_true_bin = _format_binary_labels(y_true)
                if len(pd.unique(y_true_bin)) < 2:
                    return None
                return roc_auc_score(y_true_bin, y_scores)
            if metric_name == "Precision":
                if is_multiclass:
                    return precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                try:
                    return precision_score(
                        y_true, y_pred, pos_label=pos_label, zero_division=0
                    )
                except Exception:
                    return precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
            if metric_name == "Recall":
                if is_multiclass:
                    return recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                try:
                    return recall_score(
                        y_true, y_pred, pos_label=pos_label, zero_division=0
                    )
                except Exception:
                    return recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
            if metric_name == "F1-Score":
                if is_multiclass:
                    return f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                try:
                    return f1_score(
                        y_true, y_pred, pos_label=pos_label, zero_division=0
                    )
                except Exception:
                    return f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
            if metric_name == "PR-AUC":
                if y_scores is None:
                    return None
                y_true_bin = _format_binary_labels(y_true)
                if len(pd.unique(y_true_bin)) < 2:
                    return None
                return average_precision_score(y_true_bin, y_scores)
            if metric_name == "Specificity":
                labels = pd.unique(pd.concat([y_true, y_pred], ignore_index=True))
                if len(labels) != 2:
                    return None
                if pos_label is None or pos_label not in labels:
                    pos_label = labels[1]
                neg_candidates = [lbl for lbl in labels if lbl != pos_label]
                neg_label_final = (
                    neg_label if neg_label in labels else (neg_candidates[0] if neg_candidates else None)
                )
                if neg_label_final is None:
                    return None
                cm = confusion_matrix(
                    y_true, y_pred, labels=[neg_label_final, pos_label]
                )
                if cm.shape != (2, 2):
                    return None
                tn, fp, fn, tp = cm.ravel()
                denom = tn + fp
                return (tn / denom) if denom else None
            if metric_name == "MCC":
                return matthews_corrcoef(y_true, y_pred)
        except Exception as exc:
            LOG.warning(
                "Could not compute %s for %s split: %s",
                metric_name,
                split_name,
                exc,
            )
            return None
        return None

    def _build_performance_summary_table(self):
        """
        Build a Train/Validation/Test metrics table for classification tasks.
        Returns empty string when metrics are unavailable or not applicable.
        """
        if self.task_type != "classification":
            return ""

        split_predictions = self._get_split_predictions_for_report()
        validation_best_row = None
        try:
            if isinstance(self.results, pd.DataFrame) and not self.results.empty:
                validation_best_row = self.results.iloc[0]
        except Exception:
            validation_best_row = None

        if not split_predictions and validation_best_row is None:
            return ""

        metric_names = [
            "Accuracy",
            "ROC-AUC",
            "Precision",
            "Recall",
            "F1-Score",
            "PR-AUC",
            "Specificity",
            "MCC",
        ]

        validation_column_map = {
            "Accuracy": ["Accuracy"],
            "ROC-AUC": ["ROC-AUC", "AUC"],
            "Precision": ["Precision", "Prec.", "Prec"],
            "Recall": ["Recall"],
            "F1-Score": ["F1-Score", "F1"],
            "PR-AUC": ["PR-AUC", "PR-AUC-Weighted", "PRC"],
            "Specificity": ["Specificity"],
            "MCC": ["MCC"],
        }

        def _fmt(value):
            if value is None:
                return "—"
            try:
                if isinstance(value, (float, np.floating)) and (
                    np.isnan(value) or np.isinf(value)
                ):
                    return "—"
                return f"{value:.3f}"
            except Exception:
                return str(value)

        def _validation_metric(metric_name):
            if validation_best_row is None:
                return None
            cols = validation_column_map.get(metric_name, [])
            for col in cols:
                if col in validation_best_row:
                    try:
                        return validation_best_row[col]
                    except Exception:
                        return None
            return None

        rows = []
        for metric in metric_names:
            row = [metric]
            # Train
            train_val = self._compute_metric_value(
                metric, split_predictions.get("Train"), "Train"
            )
            row.append(_fmt(train_val))

            # Validation from the cross-validation summary first row; fallback to computed CV.
            val_val = _validation_metric(metric)
            if val_val is None:
                val_val = self._compute_metric_value(
                    metric, split_predictions.get("Validation"), "Validation"
                )
            row.append(_fmt(val_val))

            # Test
            test_val = self._compute_metric_value(
                metric, split_predictions.get("Test"), "Test"
            )
            row.append(_fmt(test_val))
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Metric", "Train", "Validation", "Test"])
        return (
            "<h2>Best Model Performance</h2>"
            + '<div class="table-wrapper">'
            + df.to_html(
                index=False,
                classes=["table", "sortable", "table-perf-summary"],
            )
            + "</div>"
        )

    def _resolve_plot_callable(self, key, fig_or_fn, section):
        """
        Safely execute stored plot callables so a single failure does not
        abort the entire HTML report generation.
        """
        if fig_or_fn is None:
            return None
        try:
            return fig_or_fn() if callable(fig_or_fn) else fig_or_fn
        except Exception as exc:
            extra = ""
            if isinstance(exc, ValueError) and "Input contains NaN" in str(exc):
                extra = (
                    " (model returned NaN probabilities; "
                    "consider checking data preprocessing)"
                )
            LOG.warning(
                "Skipping %s plot '%s' due to error: %s%s",
                section,
                key,
                exc,
                extra,
            )
            return None

    @staticmethod
    def _format_parameter_label(parameter):
        """
        Convert model parameter keys like 'fit_intercept' to readable report
        labels like 'Fit Intercept'.
        """
        parts = str(parameter).replace("__", " ").replace("_", " ").split()
        return " ".join(part if part.isupper() else part.capitalize()
                        for part in parts)

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
                frac = (
                    float(v)
                    if v is not None
                    else (n_train / total_rows if total_rows else 0)
                )
                dv = f"{frac:.2f} ({n_train} rows)"
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
                dv = ", ".join(map(str, v)) if isinstance(
                    v, (list, tuple)
                ) else "None"
            elif key == "Probability Threshold":
                dv = f"{v:.2f}" if v is not None else "0.5"
            else:
                dv = v if v is not None else "None"
            setup_rows.append([key, dv])
        metric_label = self._best_model_metric_used or getattr(
            self.exp, "_fold_metric", None
        )
        if metric_label:
            setup_rows.append(["Best Model Metric", metric_label])

        df_setup = pd.DataFrame(setup_rows, columns=["Parameter", "Value"])
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
        best_model_params_df = pd.DataFrame(
            self.best_model.get_params().items(),
            columns=["Parameter", "Value"]
        )
        best_model_params_df.to_csv(
            Path(self.output_dir) / "best_model.csv", index=False
        )
        best_model_params_display_df = best_model_params_df.copy()
        best_model_params_display_df["Parameter"] = (
            best_model_params_display_df["Parameter"].map(
                self._format_parameter_label
            )
        )

        if self.tuning_results is not None:
            self.tuning_results.to_csv(
                Path(self.output_dir) / "tuning_results.csv",
                index=False
            )

        # 5) Header
        header = f"<h2>Best Model: {best_model_name}</h2>"

        # — Validation Summary & Configuration —
        val_df = self.results.copy()
        dataset_overview_html = self._build_dataset_overview()
        performance_summary_html = self._build_performance_summary_table()
        # mapping raw plot keys to user-friendly titles
        plot_title_map = {
            "learning": "Learning Curve",
            "vc": "Validation Curve",
            "calibration": "Calibration Curve",
            "dimension": "Dimensionality Reduction",
            "manifold": "t-SNE",
            "rfe": "Recursive Feature Elimination",
            "threshold": "Threshold Plot",
            "percentage_above_below": "Percentage Above vs. Below Cutoff",
            "class_report": "Per-Class Metrics",
            "pr_auc": "Precision-Recall Curve",
            "roc_auc": "Receiver Operating Characteristic AUC",
            "residuals": "Residuals Distribution",
            "error": "Prediction Error Distribution",
        }
        val_df.drop(
            columns=["TT (Ec)", "TT (Sec)"], errors="ignore", inplace=True
        )
        summary_html = (
            "<h2>Cross-Validation Validation Summary Across Candidate Models</h2>"
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

        config_html = (
            header
            + dataset_overview_html
            + performance_summary_html
            + "<h2>Experiment and Data Parameters</h2>"
            + '<div class="table-wrapper">'
            + df_setup.to_html(
                index=False,
                classes=["table", "sortable", "table-setup-params"],
            )
            + "</div>"
            # — Hyperparameters
            + "<h2>Best Model Hyperparameters</h2>"
            + '<div class="table-wrapper">'
            + best_model_params_display_df.to_html(
                index=False,
                classes=["table", "sortable", "table-hyperparams"],
            )
            + "</div>"
        )

        # choose summary plots based on task type
        if self.task_type == "classification":
            summary_plots = [
                "threshold",
                "learning",
                "calibration",
                "rfe",
                "vc",
                "dimension",
                "manifold",
                "percentage_above_below",
            ]
        else:
            summary_plots = ["learning", "vc", "parameter", "residuals"]

        for name in summary_plots:
            fig_or_fn = self.explainer_plots.pop(name, None)
            if fig_or_fn is not None:
                fig = self._resolve_plot_callable(
                    name, fig_or_fn, section="summary/explainer"
                )
                if fig is None:
                    continue
                title = plot_title_map.get(
                    name, name.replace("_", " ").title()
                )
                summary_html += (
                    "<hr>"
                    f"<h2>{title}</h2>"
                    + add_plot_to_html(fig)
                )
            elif name in self.plots:
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
                "class_report",
                "roc_auc",
                "pr_auc",
                "lift_curve",
                "cumulative_precision",
            ]
        rendered_test_plots = set()
        for key in test_order:
            fig_or_fn = self.explainer_plots.pop(key, None)
            if fig_or_fn is not None:
                fig = self._resolve_plot_callable(
                    key, fig_or_fn, section="test/explainer"
                )
                if fig is None:
                    continue
                rendered_test_plots.add(key)
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
                    "pr_auc",
                }
            ):
                if name in rendered_test_plots:
                    continue
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
        imputed_data = (
            self.imputed_training_data
            if self.imputed_training_data is not None
            else self.data
        )
        fi_analyzer = FeatureImportanceAnalyzer(
            data=imputed_data,
            target_col=self.target_col,
            task_type=self.task_type,
            output_dir=self.output_dir,
            exp=self.exp,
            best_model=self.best_model,
            max_plot_features=self.plot_feature_limit,
            processed_data=self.imputed_training_data,
            max_shap_rows=self._shap_row_cap,
        )
        fi_html = fi_analyzer.run()
        # Add a small table to show SHAP feature caps near the Best Model header.
        cap_rows = []
        if fi_analyzer.shap_total_features is not None:
            cap_rows.append(
                ("Total transformed features", fi_analyzer.shap_total_features)
            )
        if fi_analyzer.shap_used_features is not None:
            cap_rows.append(
                ("Features used in SHAP", fi_analyzer.shap_used_features)
            )
        if cap_rows:
            cap_table = (
                "<div class='table-wrapper'>"
                "<table class='table sortable table-fi-scope'>"
                "<thead><tr><th>Feature Importance Scope</th><th>Count</th></tr></thead>"
                "<tbody>"
                + "".join(
                    f"<tr><td>{label}</td><td>{value}</td></tr>"
                    for label, value in cap_rows
                )
                + "</tbody></table></div>"
            )
            feature_html += cap_table
        feature_html += fi_html

        # 6b) Explainer SHAP importances
        for key in ["shap_mean", "shap_perm"]:
            fig_or_fn = self.explainer_plots.pop(key, None)
            if fig_or_fn is not None:
                fig = self._resolve_plot_callable(
                    key, fig_or_fn, section="feature importance"
                )
                if fig is None:
                    continue
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
            fig = self._resolve_plot_callable(
                k, fig_or_fn, section="pdp"
            )
            if fig is None:
                continue
            # extract feature name
            feature = k.split("__", 1)[1]
            title = f"Partial Dependence for {feature}"
            feature_html += (
                f"<h2>{title}</h2>" + add_plot_to_html(fig)
                + add_hr_to_html()
            )
        # 7) Assemble final HTML (three tabs)
        html = get_html_template()
        html += "<h1>Tabular Learner Experiment Report</h1>"
        html += build_tabbed_html(
            summary_html,
            test_html,
            feature_html,
            explainer_html=None,
            config_html=config_html,
        )
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

    def save_dashboard(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_plots_explainer(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_tree_plots(self):
        from explainerdashboard.explainers import RandomForestExplainer
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor
        )
        from xgboost import XGBClassifier, XGBRegressor

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
        # self.save_dashboard()
