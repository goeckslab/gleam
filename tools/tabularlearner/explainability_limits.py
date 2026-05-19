import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class ExplainabilityScope:
    context: str
    total_rows: int
    used_rows: int
    total_features: int
    used_features: int
    row_cap: int
    feature_cap: int
    polynomial_features: bool = False

    @property
    def rows_capped(self):
        return self.used_rows < self.total_rows

    @property
    def features_capped(self):
        return self.used_features < self.total_features


def adaptive_row_cap(n_rows, polynomial_features=False):
    if n_rows <= 500:
        cap = n_rows
    elif n_rows <= 5000:
        cap = 500
    else:
        cap = min(1000, max(1, int(n_rows * 0.1)))
    if polynomial_features:
        cap = min(cap, 200)
    return max(0, int(cap))


def feature_cap(default_cap=30, polynomial_features=False):
    try:
        cap = int(default_cap)
    except (TypeError, ValueError):
        cap = 30
    if cap <= 0:
        cap = 30
    if polynomial_features:
        cap = min(cap, 15)
    return cap


def _model_importance_scores(model, columns):
    if model is None:
        return None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_)
        if values.ndim > 1:
            values = np.mean(np.abs(values), axis=0)
        else:
            values = np.abs(values)
    else:
        return None
    values = np.asarray(values).ravel()
    if len(values) != len(columns):
        return None
    return pd.Series(values, index=columns).abs()


def select_feature_columns(frame, model=None, max_features=30):
    if frame is None:
        return []
    columns = list(frame.columns)
    if max_features is None or max_features <= 0 or len(columns) <= max_features:
        return columns

    scores = _model_importance_scores(model, columns)
    if scores is None:
        try:
            numeric_frame = frame.select_dtypes(include=["number"])
            scores = numeric_frame.var().fillna(0).abs()
        except Exception as exc:
            LOG.warning(
                "Could not rank transformed features by variance; using input order: %s",
                exc,
            )
            return columns[:max_features]

    ranked = [col for col in scores.sort_values(ascending=False).index if col in columns]
    if len(ranked) < max_features:
        ranked.extend(col for col in columns if col not in ranked)
    return ranked[:max_features]


def limit_explainability_data(
    X,
    y=None,
    *,
    model=None,
    max_features=30,
    max_rows=None,
    random_seed=42,
    polynomial_features=False,
    cap_features=True,
    context="Explainability",
):
    X_limited = pd.DataFrame(X).copy().reset_index(drop=True)
    y_limited = None
    if y is not None:
        y_limited = pd.Series(y).reset_index(drop=True)
        if len(y_limited) != len(X_limited):
            y_limited = y_limited.iloc[: len(X_limited)].reset_index(drop=True)
            X_limited = X_limited.iloc[: len(y_limited)].reset_index(drop=True)

    total_rows, total_features = X_limited.shape
    feature_limit = feature_cap(max_features, polynomial_features)
    row_limit = (
        adaptive_row_cap(total_rows, polynomial_features)
        if max_rows is None
        else min(int(max_rows), total_rows)
    )
    row_limit = max(0, row_limit)

    if cap_features:
        selected_columns = select_feature_columns(
            X_limited,
            model=model,
            max_features=feature_limit,
        )
        if selected_columns and len(selected_columns) < total_features:
            LOG.info(
                "%s limited to top %s of %s transformed features.",
                context,
                len(selected_columns),
                total_features,
            )
            X_limited = X_limited[selected_columns]

    if total_rows > row_limit:
        LOG.info(
            "%s limited to %s of %s rows.",
            context,
            row_limit,
            total_rows,
        )
        sampled_index = X_limited.sample(row_limit, random_state=random_seed).index
        sampled_index = sorted(sampled_index.tolist())
        X_limited = X_limited.loc[sampled_index].reset_index(drop=True)
        if y_limited is not None:
            y_limited = y_limited.loc[sampled_index].reset_index(drop=True)

    scope = ExplainabilityScope(
        context=context,
        total_rows=total_rows,
        used_rows=len(X_limited),
        total_features=total_features,
        used_features=X_limited.shape[1],
        row_cap=row_limit,
        feature_cap=feature_limit,
        polynomial_features=polynomial_features,
    )
    return X_limited, y_limited, scope
