"""
Phase 5: load the trained artifacts (never re-fit), run inference and
per-instance explanations for the FastAPI endpoints.

Two rules this module enforces:
  * User text is only ever TRANSFORMed by the saved vectorizer, never
    fit. The model's weights are tied to the training vocabulary/IDF;
    refitting on one article would silently scramble what each column
    means.
  * Nothing here fabricates a result. Empty/too-short/unrecognizable
    input raises a specific error instead of returning a made-up
    confidence.

Model files are read from MODEL_DIR only, and the model name from the
caller is mapped through a fixed whitelist -- it is never used to build
a file path. joblib uses pickle under the hood, so it must only ever
load files this project wrote itself.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import joblib
import numpy as np

from config import LABEL_NAMES, MODEL_DIR
from src.preprocessing import normalize_whitespace, strip_source_artifacts

logger = logging.getLogger(__name__)

# Keys match the stable model identifiers in metrics.json. File names
# must match the paths written by src/train.py.
VECTORIZER_FILE = "vectorizer.joblib"
MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "multinomial_nb": "multinomial_nb.joblib",
}
METRICS_FILE = "metrics.json"

# Inputs with fewer words than this (after cleaning) carry too little
# signal for the confidence number to mean anything.
MIN_WORDS = 3

DEFAULT_TOP_N = 10

EXPLANATION_NOTE = (
    "These are the words and phrases that pushed the MODEL toward or away from its "
    "predicted label ({label}). They describe how the model reached its output, NOT "
    "evidence that the article is true or false."
)


# --------------------------------------------------------------------------
# Errors -- each maps to a clean HTTP response in main.py
# --------------------------------------------------------------------------
class PredictionError(Exception):
    """Base class; str(exc) is always safe to show to an API client."""


class ModelNotTrainedError(PredictionError):
    """Model artifacts are missing from MODEL_DIR."""


class ModelLoadError(PredictionError):
    """Artifacts exist but could not be loaded."""


class InvalidModelError(PredictionError):
    """Requested model name is not one we serve."""


class EmptyInputError(PredictionError):
    """Text is empty or whitespace only."""


class TextTooShortError(PredictionError):
    """Text has fewer than MIN_WORDS words."""


class NoRecognizedWordsError(PredictionError):
    """None of the words are in the model's vocabulary."""


# --------------------------------------------------------------------------
# Artifact loading (lazy, cached)
# --------------------------------------------------------------------------
_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()

_NOT_TRAINED_HINT = "Run `python -m src.train` from the ml-service directory to train and save the models."


def _load_artifact(filename: str) -> Any:
    path = os.path.join(MODEL_DIR, filename)
    with _cache_lock:
        if path in _cache:
            return _cache[path]
        if not os.path.isfile(path):
            raise ModelNotTrainedError(f"Model file '{filename}' not found. {_NOT_TRAINED_HINT}")
        try:
            artifact = joblib.load(path)
        except Exception as exc:  # unpickling can fail in many ways (version mismatch, corruption)
            logger.exception("Failed to load %s", path)
            raise ModelLoadError(
                f"Model file '{filename}' could not be loaded (corrupt, or saved with different "
                f"library versions). Re-train with `python -m src.train`."
            ) from exc
        _cache[path] = artifact
        return artifact


def available_models() -> list[str]:
    """Model names whose files (and the shared vectorizer) exist on disk."""
    if not os.path.isfile(os.path.join(MODEL_DIR, VECTORIZER_FILE)):
        return []
    return [name for name, fname in MODEL_FILES.items() if os.path.isfile(os.path.join(MODEL_DIR, fname))]


def metrics_path() -> str:
    return os.path.join(MODEL_DIR, METRICS_FILE)


def _feature_names(vectorizer) -> np.ndarray:
    with _cache_lock:
        names = _cache.get("__feature_names__")
        if names is None:
            names = vectorizer.get_feature_names_out()
            _cache["__feature_names__"] = names
        return names


# --------------------------------------------------------------------------
# Shared input handling
# --------------------------------------------------------------------------
def clean_input(text: str) -> str:
    """
    The same cleaning steps training applied to article text: strip
    wire-service markers, then collapse whitespace. (Training did this to
    the body before joining the title; a single pasted blob is treated
    as the body.)
    """
    stripped, _ = strip_source_artifacts(text)
    return normalize_whitespace(stripped)


def _prepare(text: str, model_name: str):
    """Validate input and return (model, vectorizer, feature_row) ready for inference."""
    if model_name not in MODEL_FILES:
        raise InvalidModelError(
            f"Unknown model '{model_name}'. Valid models: {', '.join(MODEL_FILES)}."
        )
    if not isinstance(text, str) or not text.strip():
        raise EmptyInputError("Text is empty. Paste a headline or article to analyze.")

    cleaned = clean_input(text)
    n_words = len(cleaned.split())
    if n_words < MIN_WORDS:
        raise TextTooShortError(
            f"Text is too short to analyze ({n_words} word(s)). Provide at least {MIN_WORDS} words."
        )

    vectorizer = _load_artifact(VECTORIZER_FILE)
    model = _load_artifact(MODEL_FILES[model_name])

    features = vectorizer.transform([cleaned])  # transform ONLY -- never fit on user input
    if features.nnz == 0:
        # An all-zero row would make the model fall back on its intercept and
        # return a confident-looking answer based on no evidence at all.
        raise NoRecognizedWordsError(
            "None of the words in this text are in the model's vocabulary, so there is nothing to analyze."
        )
    return model, vectorizer, features


def _predicted_class(model, features) -> tuple[int, int, float]:
    """Returns (index into model.classes_, class label id, probability of that class)."""
    proba = model.predict_proba(features)[0]
    idx = int(np.argmax(proba))
    return idx, int(model.classes_[idx]), float(proba[idx])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def predict(text: str, model_name: str) -> dict[str, Any]:
    """
    Returns {"label": "REAL"|"FAKE", "probability": float, "model": model_name}.

    `probability` is the model's confidence in its predicted label. It is NOT
    the probability that the story is factually true.
    """
    model, _, features = _prepare(text, model_name)
    _, label_id, prob = _predicted_class(model, features)
    return {"label": LABEL_NAMES[label_id], "probability": round(prob, 4), "model": model_name}


def explain(text: str, model_name: str, top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
    """
    Per-word contributions for the words actually present in `text`.

    contribution = (weight favouring the predicted class) x (word's TF-IDF value)

      * Logistic Regression: weight = the coefficient, sign-flipped if the
        predicted class is the one the coefficient points away from.
      * Multinomial NB: weight = log P(word | predicted) - log P(word | other),
        which is exactly the per-word term in NB's decision rule.

    top_positive pushed toward the predicted label, top_negative pushed away
    from it. Words the vectorizer doesn't know are ignored, as in predict().
    """
    model, vectorizer, features = _prepare(text, model_name)
    pred_idx, label_id, _ = _predicted_class(model, features)

    if model_name == "logistic_regression":
        # Binary LR has one coefficient row, pointing toward classes_[1].
        coef = model.coef_[0]
        weights = coef if pred_idx == 1 else -coef
    else:  # multinomial_nb -- the only other name _prepare lets through
        log_prob = model.feature_log_prob_
        weights = log_prob[pred_idx] - log_prob[1 - pred_idx]

    present = features.indices  # columns with a non-zero TF-IDF value in this text
    contributions = features.data * weights[present]
    names = _feature_names(vectorizer)[present]

    order = np.argsort(contributions)
    top_negative = [
        {"word": str(names[i]), "contribution": round(float(contributions[i]), 4)}
        for i in order
        if contributions[i] < 0
    ][:top_n]
    top_positive = [
        {"word": str(names[i]), "contribution": round(float(contributions[i]), 4)}
        for i in order[::-1]
        if contributions[i] > 0
    ][:top_n]

    return {
        "model": model_name,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "note": EXPLANATION_NOTE.format(label=LABEL_NAMES[label_id]),
    }
