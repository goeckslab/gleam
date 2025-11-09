"""Feature importance visualization utilities."""

import pandas as pd

def build_feature_importance_html(predictor, df_train: pd.DataFrame, label_column: str) -> str:
    """Feature importance is not currently available for the MultiModal workflow."""
    return (
        "<p><em>Feature importance visualization is not supported for the current "
        "MultiModal workflow.</em></p>"
    )
