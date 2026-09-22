"""
NewsCheck: Streamlit thin client.

All data comes from the Spring Boot backend through api_client.py. This file
contains no ML logic and never touches the Python service or any model file.

Run from the frontend/ directory:  streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st

import api_client
import charts
from api_client import MODEL_LABELS, BackendError

MAX_TEXT_CHARS = 20000  # same limit the backend enforces

st.set_page_config(page_title="NewsCheck", page_icon="📰", layout="wide")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def dig(data: Any, *keys: str) -> Any:
    """data[k1][k2]... or None if any level is missing. Lets the UI skip absent fields."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def fmt_int(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "—"


def fmt_time(iso: Optional[str]) -> str:
    """ISO-8601 UTC timestamp -> local 'YYYY-MM-DD HH:MM'."""
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def model_name(key: Optional[str]) -> str:
    return MODEL_LABELS.get(key or "", key or "—")


def show_backend_error(err: BackendError) -> None:
    """Friendly message for a failed backend call; never a traceback."""
    if err.kind == "invalid_input":
        st.warning(err.message)
    elif err.kind == "ml_down":
        st.error(err.message)
        st.caption(
            "The backend is running, but the Python analysis service behind it isn't ready. "
            "Start it with `uvicorn main:app` from `ml-service/` (run `python -m src.train` first "
            "if the models haven't been trained)."
        )
    elif err.kind == "unreachable":
        st.error(err.message)
        st.caption("Start the Spring Boot backend (port 8080) and try again.")
    else:
        st.error(err.message)


# --------------------------------------------------------------------------
# Backend data (fetched once per run, failures kept as values not exceptions)
# --------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def _cached_metrics() -> dict[str, Any]:
    return api_client.metrics()  # exceptions aren't cached, so a failure is retried next run


def load_health() -> tuple[Optional[dict[str, Any]], Optional[BackendError]]:
    try:
        return api_client.health(), None
    except BackendError as err:
        return None, err


def load_metrics() -> tuple[Optional[dict[str, Any]], Optional[BackendError]]:
    try:
        return _cached_metrics(), None
    except BackendError as err:
        return None, err


health, health_error = load_health()
backend_up = health_error is None or health_error.kind not in ("unreachable", "timeout")
# Skip the other calls when the backend is plainly down, so the page stays snappy.
metrics, metrics_error = load_metrics() if backend_up else (None, health_error)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def status_line(state: str, text: str) -> None:
    dot = {"ok": ":green[●]", "warn": ":orange[●]", "bad": ":red[●]"}[state]
    st.markdown(f"{dot} {text}")


def render_status() -> None:
    if health_error is not None:
        status_line("bad", "**Backend:** not reachable")
        st.caption(health_error.message)
        return

    status_line("ok", "**Backend:** running")
    ml = health.get("ml_service") or {}
    if ml.get("status") != "UP":
        status_line("bad", "**ML service:** down")
        if ml.get("message"):
            st.caption(ml["message"])
        return

    status_line("ok", "**ML service:** running")
    if ml.get("trained"):
        available = ", ".join(model_name(m) for m in ml.get("models_available") or [])
        status_line("ok", "**Models:** trained")
        if available:
            st.caption(f"Available: {available}")
    else:
        status_line("warn", "**Models:** not trained")
        st.caption(ml.get("message") or "Run `python -m src.train` in ml-service.")


def render_dataset_summary() -> None:
    dataset, split = dig(metrics, "metadata", "dataset"), dig(metrics, "metadata", "split")
    if not dataset and not split:
        st.caption("Not available yet. It appears once the backend can reach a trained ML service.")
        return
    if dataset:
        st.markdown(f"**{fmt_int(dataset.get('total_rows'))}** articles after cleaning")
        st.caption(f"REAL {fmt_int(dataset.get('real_count'))} · FAKE {fmt_int(dataset.get('fake_count'))}")
    if split:
        st.caption(f"Train {fmt_int(split.get('train_size'))} · Test {fmt_int(split.get('test_size'))}")


with st.sidebar:
    st.title("NewsCheck")
    st.caption(
        "Paste a headline or article and see whether a machine-learning model finds its writing "
        "closer to REAL or FAKE news from its training data. It does not check facts."
    )

    model_key = st.selectbox(
        "Model", options=list(MODEL_LABELS), format_func=model_name, key="model_key",
        help="Both models use the same text features; they differ in how they weigh them.",
    )

    st.divider()
    st.subheader("System status")
    render_status()

    st.divider()
    st.subheader("Training data")
    render_dataset_summary()


# --------------------------------------------------------------------------
# Main page: input
# --------------------------------------------------------------------------
st.title("NewsCheck — Fake News Classifier")
st.caption("Educational ML demo. A prediction here is not a verdict on whether a story is true.")

text = st.text_area(
    "Paste a headline or article",
    height=200,
    max_chars=MAX_TEXT_CHARS,
    placeholder="Paste the text you want to analyze…",
    key="article_text",
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please paste a headline or article first.")
    else:
        with st.spinner("Analyzing…"):
            try:
                st.session_state["result"] = api_client.analyze(text, model_key)
            except BackendError as err:
                st.session_state.pop("result", None)  # don't leave an old result under a new error
                show_backend_error(err)


# --------------------------------------------------------------------------
# Main page: result + explanation
# --------------------------------------------------------------------------
def render_result(result: dict[str, Any]) -> None:
    label, prob = result["label"], result["probability"]
    color = "green" if label == "REAL" else "red"

    with st.container(border=True):
        st.markdown(f"## :{color}[Model prediction: {label}-like text]")
        st.metric("Model confidence", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Model used: {model_name(result.get('model'))}. Confidence is how sure the model is of "
            "its own label. It is **not** the probability that the story is factually true."
        )

    st.warning(f"**{result.get('disclaimer') or 'ML prediction, not fact-checking'}.** "
               "The model recognizes patterns in how text is written; it cannot verify claims.")

    st.subheader("Why did the model make this prediction?")
    explanation = result.get("explanation") or {}
    positive = explanation.get("top_positive") or []
    negative = explanation.get("top_negative") or []
    figure = charts.explanation_chart(positive, negative)
    if figure is None:
        st.info("No word-level detail is available for this text.")
    else:
        st.caption(
            f"Words and phrases from your text that pushed the model toward its prediction "
            f"({result['label']}, blue) or away from it (orange)."
        )
        st.plotly_chart(figure, width="stretch")
        if not negative:
            st.caption("No words pushed against this prediction.")
    st.info(
        explanation.get("note")
        or "These are the words that pushed the model, not evidence that the article is true or false."
    )


result = st.session_state.get("result")
if isinstance(result, dict) and "label" in result and "probability" in result:
    render_result(result)


# --------------------------------------------------------------------------
# Model performance dashboard
# --------------------------------------------------------------------------
def render_selected_model_tab() -> None:
    model = dig(metrics, "models", model_key)
    if not model:
        st.info("No stored performance numbers for this model.")
        return

    test_metrics = model.get("test_metrics") or {}
    scores = charts.extract_scores(test_metrics)
    test_size = dig(metrics, "metadata", "split", "test_size")
    st.caption(
        f"{model.get('display_name', model_name(model_key))}, scored on "
        f"{fmt_int(test_size)} held-out test articles the model never saw during training."
    )

    columns = st.columns(len(scores))
    for column, (name, value) in zip(columns, scores.items()):
        if value is None:
            column.metric(name, "—")
        else:
            column.metric(name, f"{value:.4f}" if name == "ROC-AUC" else f"{value:.2%}")
    if test_metrics.get("roc_auc_note"):
        st.caption(f"ROC-AUC note: {test_metrics['roc_auc_note']}")

    leakage = dig(metrics, "metadata", "leakage") or {}
    if model_key in (leakage.get("models_above_threshold") or []):
        stripped = leakage.get("wire_service_markers_stripped") or {}
        threshold = leakage.get("accuracy_warning_threshold")
        accuracy = scores.get("Accuracy")
        st.warning(
            "**Interpret with caution.** "
            + (f"This model's test accuracy ({accuracy:.1%}) is above the {threshold:.0%} warning threshold. "
               if accuracy is not None and threshold is not None else "")
            + "This dataset has known source-style patterns: wire-service markers such as “(Reuters)” "
            f"were found (and removed) in {fmt_int(stripped.get('REAL'))} REAL articles but only "
            f"{fmt_int(stripped.get('FAKE'))} FAKE ones. Very high scores can partly reflect quirks of the "
            "dataset rather than real-world ability."
        )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion matrix** (test set)")
        figure = charts.confusion_matrix_chart(test_metrics.get("confusion_matrix") or {})
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
            st.caption("Rows are the true class, columns are what the model predicted.")
        else:
            st.caption("Not available.")
    with right:
        st.markdown("**Per-class results**")
        per_class = test_metrics.get("per_class") or {}
        if per_class:
            table = pd.DataFrame(
                [
                    {
                        "Class": name,
                        "Precision": stats.get("precision"),
                        "Recall": stats.get("recall"),
                        "F1": stats.get("f1"),
                        "Articles": stats.get("support"),
                    }
                    for name, stats in per_class.items()
                ]
            )
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Precision": st.column_config.NumberColumn(format="%.3f"),
                    "Recall": st.column_config.NumberColumn(format="%.3f"),
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                    "Articles": st.column_config.NumberColumn(format="%d"),
                },
            )
        else:
            st.caption("Not available.")


def render_comparison_tab() -> None:
    models = metrics.get("models") or {}
    order = dig(metrics, "comparison", "ranking_by_macro_f1") or list(models)
    figure = charts.comparison_chart(models, order)
    if figure is None:
        st.info("No model comparison is available.")
        return

    st.plotly_chart(figure, width="stretch")
    if any(m.get("is_baseline") for m in models.values()):
        st.caption(
            "The grey Dummy baseline always guesses the most common class. It is a floor to beat, "
            "not a real model."
        )

    rows = []
    for key in order:
        model = models.get(key)
        if model:
            rows.append({"Model": model.get("display_name", key),
                         **charts.extract_scores(model.get("test_metrics") or {})})
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={name: st.column_config.NumberColumn(format="%.4f") for name, _ in charts.SCORE_FIELDS},
        )


def render_training_data_tab() -> None:
    dataset = dig(metrics, "metadata", "dataset") or {}
    split = dig(metrics, "metadata", "split") or {}
    tiles = [
        ("Articles (after cleaning)", dataset.get("total_rows")),
        ("REAL", dataset.get("real_count")),
        ("FAKE", dataset.get("fake_count")),
        ("Training set", split.get("train_size")),
        ("Test set", split.get("test_size")),
    ]
    tiles = [(label, value) for label, value in tiles if value is not None]
    if not tiles:
        st.info("No dataset information is available.")
        return

    for column, (label, value) in zip(st.columns(len(tiles)), tiles):
        column.metric(label, fmt_int(value))

    cleaning = dataset.get("cleaning") or {}
    if dataset.get("rows_in") is not None:
        st.caption(
            f"Started with {fmt_int(dataset['rows_in'])} rows; removed "
            f"{fmt_int(cleaning.get('dropped_exact_duplicates'))} exact and "
            f"{fmt_int(cleaning.get('dropped_near_duplicates'))} near-duplicate articles."
        )
    if split.get("test_fraction") is not None:
        st.caption(f"{split['test_fraction']:.0%} of articles were held out as the test set (stratified by class).")
    trained_at = dig(metrics, "metadata", "trained_at_utc")
    if trained_at:
        st.caption(f"Models trained: {fmt_time(trained_at)}")


st.divider()
st.header("Model performance")
if metrics is None:
    st.info(
        "Performance numbers aren't available right now"
        + (f": {metrics_error.message}" if metrics_error else ".")
    )
else:
    tab_selected, tab_compare, tab_data = st.tabs(["Selected model", "Model comparison", "Training data"])
    with tab_selected:
        render_selected_model_tab()
    with tab_compare:
        render_comparison_tab()
    with tab_data:
        render_training_data_tab()


# --------------------------------------------------------------------------
# Recent history (fetched last so a just-finished analysis appears in it)
# --------------------------------------------------------------------------
st.divider()
st.header("Recent analyses")
if not backend_up:
    st.info("History is unavailable while the backend is unreachable.")
else:
    try:
        rows = api_client.history(10)
    except BackendError as err:
        st.info(f"History is unavailable right now: {err.message}")
    else:
        if not rows:
            st.caption("Nothing analyzed yet. Your analyses will appear here.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "When": fmt_time(row["created_at"]),
                            "Prediction": f"{row['label']}-like" if row.get("label") else "",
                            "Confidence": (row["probability"] or 0) * 100,
                            "Model": model_name(row.get("model")),
                            "Text": row["text_snippet"],
                        }
                        for row in rows
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Confidence": st.column_config.NumberColumn(format="%.1f%%"),
                    "Text": st.column_config.TextColumn(width="large"),
                },
            )
