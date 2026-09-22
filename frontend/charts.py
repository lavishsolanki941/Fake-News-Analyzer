"""
Plotly figure builders for the Streamlit app. Every function takes data
exactly as the backend returned it and returns None when the pieces it
needs aren't there, so the UI can simply skip a chart instead of showing
made-up numbers.
"""
from __future__ import annotations

from typing import Any, Optional

import plotly.express as px
import plotly.graph_objects as go

# Explanation bars use blue/orange on purpose: green/red already mean REAL/FAKE
# elsewhere in the app, and "pushed toward/away" is a different idea.
TOWARD_COLOR = "#3b82c4"
AWAY_COLOR = "#e08a2c"
_REAL_MODEL_COLORS = ["#3b82c4", "#e08a2c"]
BASELINE_COLOR = "#9aa0a6"  # the Dummy baseline is always grey: it's a floor, not a contender

# (label shown to the user, how to read it from a model's test_metrics)
SCORE_FIELDS = [
    ("Accuracy", lambda tm: tm.get("accuracy")),
    ("Precision (macro)", lambda tm: (tm.get("macro_avg") or {}).get("precision")),
    ("Recall (macro)", lambda tm: (tm.get("macro_avg") or {}).get("recall")),
    ("F1 (macro)", lambda tm: (tm.get("macro_avg") or {}).get("f1")),
    ("ROC-AUC", lambda tm: tm.get("roc_auc")),
]


def extract_scores(test_metrics: dict[str, Any]) -> dict[str, Optional[float]]:
    """{'Accuracy': 0.98, ..., 'ROC-AUC': None} for one model's test_metrics."""
    return {label: read(test_metrics) for label, read in SCORE_FIELDS}


def explanation_chart(
    top_positive: list[dict[str, Any]], top_negative: list[dict[str, Any]]
) -> Optional[go.Figure]:
    """Horizontal bars: words that pushed toward (right, blue) vs away (left, orange)."""
    rows = [(w["word"], w["contribution"], True) for w in top_positive or []]
    rows += [(w["word"], w["contribution"], False) for w in top_negative or []]
    if not rows:
        return None

    rows.sort(key=lambda r: r[1], reverse=True)  # biggest push toward at the top
    fig = go.Figure(
        go.Bar(
            x=[r[1] for r in rows],
            y=[r[0] for r in rows],
            orientation="h",
            marker_color=[TOWARD_COLOR if r[2] else AWAY_COLOR for r in rows],
            text=[f"{r[1]:+.3f}" for r in rows],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Push toward (+) or away from (−) the predicted label", zeroline=True)
    fig.update_layout(
        height=max(220, 30 * len(rows) + 90),
        margin=dict(l=10, r=40, t=10, b=10),
        showlegend=False,
    )
    return fig


def confusion_matrix_chart(cm: dict[str, Any]) -> Optional[go.Figure]:
    """Heatmap with counts in each cell. Rows = actual class, columns = predicted class."""
    matrix, labels = cm.get("matrix"), cm.get("labels")
    if not matrix or not labels or len(matrix) != len(labels):
        return None

    fig = px.imshow(
        matrix,
        x=[f"Predicted {name}" for name in labels],
        y=[f"Actual {name}" for name in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(hovertemplate="%{y}, %{x}: %{z} articles<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def comparison_chart(models: dict[str, dict[str, Any]], order: list[str]) -> Optional[go.Figure]:
    """Grouped bars: one group per metric, one bar per model (baseline in grey)."""
    fig = go.Figure()
    real_model_colors = iter(_REAL_MODEL_COLORS)
    for key in order:
        model = models.get(key)
        if not model:
            continue
        scores = {k: v for k, v in extract_scores(model.get("test_metrics") or {}).items() if v is not None}
        if not scores:
            continue
        color = BASELINE_COLOR if model.get("is_baseline") else next(real_model_colors, BASELINE_COLOR)
        fig.add_bar(
            name=model.get("display_name", key),
            x=list(scores),
            y=list(scores.values()),
            marker_color=color,
            text=[f"{v:.3f}" for v in scores.values()],
            textposition="outside",
            hovertemplate="%{fullData.name}<br>%{x}: %{y:.4f}<extra></extra>",
        )
    if not fig.data:
        return None

    fig.update_layout(
        barmode="group",
        height=420,
        yaxis=dict(range=[0, 1.1], title="Score (test set)"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
