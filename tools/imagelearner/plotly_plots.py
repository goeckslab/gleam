import json
from typing import List, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def build_classification_plots(
    test_stats_path: str,
    training_stats_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Read Ludwig’s test_statistics.json and build three interactive Plotly panels:
      - Confusion Matrix
      - ROC-AUC
      - Classification Report Heatmap

    Returns a list of dicts, each with:
      {
        "title": <plot title>,
        "html":  <HTML fragment for embedding>
      }
    """
    # --- Load test stats ---
    with open(test_stats_path, "r") as f:
        test_stats = json.load(f)
    label_stats = test_stats["label"]

    # --- Load test ROC-AUC from training_statistics.json if provided ---
    test_aucs: List[float] = []
    if training_stats_path:
        try:
            with open(training_stats_path, "r") as f:
                stats = json.load(f)
            test_aucs = (
                stats.get("test", {})
                     .get("label", {})
                     .get("roc_auc", [])
            )
        except Exception:
            test_aucs = []

    # common sizing
    cell = 40
    n_classes = len(label_stats["confusion_matrix"])
    side_px = max(cell * n_classes + 200, 600)
    common_cfg = {"displayModeBar": True, "scrollZoom": True}

    plots: List[Dict[str, str]] = []

    # 0) Confusion Matrix
    cm = np.array(label_stats["confusion_matrix"], dtype=int)
    labels = label_stats.get("labels", [str(i) for i in range(cm.shape[0])])
    total = cm.sum()

    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig_cm.update_traces(xgap=2, ygap=2)
    fig_cm.update_layout(
        title=dict(text="Confusion Matrix", x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Observed",
        yaxis_autorange="reversed",
        width=side_px,
        height=side_px,
        margin=dict(t=100, l=80, r=80, b=80),
    )

    # annotate counts and percentages
    mval = cm.max() if cm.size else 0
    thresh = mval / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            pct = (v / total * 100) if total > 0 else 0
            color = "white" if v > thresh else "black"
            fig_cm.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"<b>{v}</b>",
                showarrow=False,
                font=dict(color=color, size=14),
                xanchor="center",
                yanchor="bottom",
                yshift=2,
            )
            fig_cm.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{pct:.1f}%",
                showarrow=False,
                font=dict(color=color, size=13),
                xanchor="center",
                yanchor="top",
                yshift=-2,
            )

    plots.append({
        "title": "Confusion Matrix",
        "html": pio.to_html(
            fig_cm,
            full_html=False,
            include_plotlyjs="cdn",
            config=common_cfg
        )
    })

#    # 1) ROC-AUC (Test data)
#    if test_aucs:
#        evals = list(range(1, len(test_aucs) + 1))
#
#        fig_auc = go.Figure()
#
#        # ROC-AUC curve
#        fig_auc.add_trace(go.Scatter(
#            x=evals,
#            y=test_aucs,
#            mode="lines+markers",
#            name="ROC-AUC",
#            line=dict(color="royalblue", width=3),
#            marker=dict(size=8, color="royalblue"),
#        ))
#
#        # Add random baseline
#        fig_auc.add_trace(go.Scatter(
#            x=[evals[0], evals[-1]],
#            y=[0.5, 0.5],
#            mode="lines",
#            name="Chance level (0.5)",
#            line=dict(color="gray", dash="dash"),
#        ))
#
#        fig_auc.update_layout(
#            title="<b>ROC-AUC Over Evaluation</b>",
#            xaxis_title="<b>Evaluation</b>",
#            yaxis_title="<b>True Positive Rate</b>",
#            font=dict(size=14),
#            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='gray'),
#            plot_bgcolor="white",
#            width=side_px,
#            height=side_px,
#            margin=dict(t=80, l=80, r=80, b=80),
#            xaxis=dict(showgrid=True, zeroline=False),
#            yaxis=dict(showgrid=True, zeroline=False, range=[0, 1.05])
#        )
#
#        plots.append({
#            "title": "ROC-AUC",
#            "html": pio.to_html(
#                fig_auc,
#                full_html=False,
#                include_plotlyjs=False,
#                config=common_cfg
#            )
#        })

    # 2) Classification Report Heatmap
    pcs = label_stats.get("per_class_stats", {})
    if pcs:
        classes = list(pcs.keys())
        metrics = ["precision", "recall", "f1_score"]
        z, txt = [], []
        for c in classes:
            row, trow = [], []
            for m in metrics:
                val = pcs[c].get(m, 0)
                row.append(val)
                trow.append(f"{val:.2f}")
            z.append(row)
            txt.append(trow)

        fig_cr = go.Figure(
            go.Heatmap(
                z=z,
                x=metrics,
                y=[str(c) for c in classes],
                text=txt,
                texttemplate="%{text}",
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Value"),
            )
        )
        fig_cr.update_layout(
            title="Classification Report",
            xaxis_title="",
            yaxis_title="Class",
            width=side_px,
            height=side_px,
            margin=dict(t=80, l=80, r=80, b=80),
        )
        plots.append({
            "title": "Classification Report",
            "html": pio.to_html(
                fig_cr,
                full_html=False,
                include_plotlyjs=False,
                config=common_cfg
            )
        })

    return plots
