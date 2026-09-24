"""
Tests for src.predict: empty/too-short/no-known-word input handling,
prediction output format, and model+vectorizer artifact loading -- including
the clear error raised when artifacts are missing or corrupt. Models here
are tiny ones fit on a handful of synthetic sentences (see conftest.py's
trained_model_dir fixture) inside a pytest temp directory; the project's
real ml-service/models artifacts are never touched.
"""
from __future__ import annotations

import joblib
import pytest

from src import predict as predictor


# --------------------------------------------------------------------------
# Empty / too-short input handling
# --------------------------------------------------------------------------
def test_predict_raises_on_empty_text(use_trained_model_dir):
    with pytest.raises(predictor.EmptyInputError):
        predictor.predict("", "logistic_regression")


def test_predict_raises_on_whitespace_only_text(use_trained_model_dir):
    with pytest.raises(predictor.EmptyInputError):
        predictor.predict("   \n\t  ", "logistic_regression")


def test_predict_raises_on_too_short_text(use_trained_model_dir):
    # "hi there" is 2 words; MIN_WORDS is 3.
    with pytest.raises(predictor.TextTooShortError):
        predictor.predict("hi there", "logistic_regression")


def test_predict_raises_on_unknown_model(use_trained_model_dir):
    with pytest.raises(predictor.InvalidModelError):
        predictor.predict("this is a perfectly normal length sentence", "some_other_model")


def test_predict_raises_when_no_words_recognized(use_trained_model_dir):
    # Enough words to pass the length check, but none are in the tiny vocab.
    with pytest.raises(predictor.NoRecognizedWordsError):
        predictor.predict("zzqx wibblequonk fribbleton splargh", "logistic_regression")


# --------------------------------------------------------------------------
# Prediction output format
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ["logistic_regression", "multinomial_nb"])
def test_predict_output_shape(use_trained_model_dir, model_name):
    result = predictor.predict("senate committee approves new budget plan", model_name)

    assert set(result.keys()) == {"label", "probability", "model"}
    assert result["label"] in ("REAL", "FAKE")
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model"] == model_name


def test_explain_output_shape(use_trained_model_dir):
    result = predictor.explain("shocking secret miracle cure revealed today", "logistic_regression")

    assert set(result.keys()) == {"model", "top_positive", "top_negative", "note"}
    assert result["model"] == "logistic_regression"
    for bucket in ("top_positive", "top_negative"):
        assert isinstance(result[bucket], list)
        for item in result[bucket]:
            assert set(item.keys()) == {"word", "contribution"}


# --------------------------------------------------------------------------
# Model + vectorizer loading
# --------------------------------------------------------------------------
def test_available_models_empty_dir_returns_empty_list(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_DIR", str(tmp_path))
    assert predictor.available_models() == []


def test_available_models_lists_present_models(use_trained_model_dir):
    assert set(predictor.available_models()) == {"logistic_regression", "multinomial_nb"}


def test_predict_raises_not_trained_when_artifacts_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_DIR", str(tmp_path))

    with pytest.raises(predictor.ModelNotTrainedError, match="python -m src.train"):
        predictor.predict("this is a perfectly normal length sentence", "logistic_regression")


def test_predict_raises_model_load_error_on_corrupt_artifact(monkeypatch, tmp_path, trained_model_dir):
    # A valid vectorizer, but a model file that isn't a real joblib pickle --
    # simulates corruption / a version mismatch, as opposed to a missing file.
    joblib.dump(joblib.load(trained_model_dir / "vectorizer.joblib"), tmp_path / "vectorizer.joblib")
    (tmp_path / "logistic_regression.joblib").write_bytes(b"not a real pickle")

    monkeypatch.setattr(predictor, "MODEL_DIR", str(tmp_path))
    predictor._cache.clear()

    with pytest.raises(predictor.ModelLoadError):
        predictor.predict("this is a perfectly normal length sentence", "logistic_regression")
