"""Feature importance visualization utilities."""

import html
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

def build_feature_importance_html(predictor, df_train: pd.DataFrame, label_column: str) -> str:
    """Build feature importance visualization and explanation.
    
    For MultiModalPredictor, explains that permutation importance is not supported.
    For TabularPredictor, shows feature importance from the predictor.
    """
    # For MultiModalPredictor
    from autogluon.multimodal import MultiModalPredictor
    if isinstance(predictor, MultiModalPredictor):
        return (
            "<p><em>Feature importance visualization is not supported for MultiModalPredictor. "
            "This functionality is only available for tabular-only models.</em></p>"
        )
    
    # For TabularPredictor
    try:
        # Get feature importance scores
        importance_scores = None
        if hasattr(predictor, "feature_importance"):
            importance_scores = predictor.feature_importance(df_train)
            
        # Format scores into a DataFrame if not already
        if isinstance(importance_scores, dict):
            importance_df = pd.DataFrame(
                [(k, v) for k, v in importance_scores.items()],
                columns=["feature", "importance"]
            )
        elif isinstance(importance_scores, pd.DataFrame):
            importance_df = importance_scores.copy()
        else:
            return "<p>Feature importance information is not available for this model.</p>"
            
        # Sort by importance descending
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        # Build table HTML
        rows = []
        for _, row in importance_df.iterrows():
            feat = row["feature"]
            imp = float(row["importance"])
            rows.append(f"<tr><td>{html.escape(str(feat))}</td><td>{imp:.4f}</td></tr>")
            
        if not rows:
            return "<p>No feature importance scores available.</p>"
            
        html_out = [
            "<table class='feature-importance'>",
            "<thead><tr><th>Feature</th><th>Importance Score</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>"
        ]
        return "\n".join(html_out)
        
    except Exception as e:
        return f"<p>Error computing feature importance: {str(e)}</p>"