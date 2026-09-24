"""
Shared pytest fixtures for the ml-service test suite.

Everything here builds small SYNTHETIC data on the fly, in a pytest-managed
temp directory. Nothing in this test suite reads ml-service/data/*.csv or
ml-service/models/*.joblib, and nothing makes a network call -- that's a
deliberate rule so the tests stay fast, reproducible, and safe to run
anywhere (including CI) without the real (large) dataset present.
"""
from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src import predict as predictor


@pytest.fixture(autouse=True)
def _clear_predictor_cache():
    """src/predict.py caches loaded artifacts in memory; start each test clean."""
    predictor._cache.clear()
    yield
    predictor._cache.clear()


# --------------------------------------------------------------------------
# Synthetic full-article rows, for data_loader / preprocessing tests.
# Shaped like the real Fake.csv/True.csv (title, text, subject, date) so the
# code under test sees the same columns it does in production.
# --------------------------------------------------------------------------
FAKE_ROWS = [
    {
        "title": "You Won't Believe This Shocking Secret",
        "text": "Click here now for the shocking secret they dont want you to know about at all.",
        "subject": "News",
        "date": "January 1, 2020",
    },
    {
        "title": "Miracle Cure Doctors Hate",
        "text": "This one weird trick cures everything instantly according to anonymous sources close to the matter.",
        "subject": "News",
        "date": "January 2, 2020",
    },
    {
        "title": "Aliens Confirmed By Secret Agency",
        "text": "Anonymous insiders claim aliens are real and the government is hiding it from everyone alive today.",
        "subject": "News",
        "date": "January 3, 2020",
    },
    {
        "title": "Celebrity Says Shocking Thing",
        "text": "Sources say a celebrity said something shocking that will completely blow your mind forever.",
        "subject": "News",
        "date": "January 4, 2020",
    },
]

REAL_ROWS = [
    {
        "title": "Senate Passes Budget Bill",
        "text": "WASHINGTON (Reuters) - The Senate passed a budget bill on Tuesday after months of negotiation.",
        "subject": "politicsNews",
        "date": "January 1, 2020",
    },
    {
        "title": "Central Bank Holds Interest Rates",
        "text": "LONDON (Reuters) - The central bank held interest rates steady on Wednesday, citing inflation data.",
        "subject": "worldnews",
        "date": "January 2, 2020",
    },
    {
        "title": "City Council Approves New Budget",
        "text": "NEW YORK (Reuters) - The city council approved a new annual budget after a long public session.",
        "subject": "politicsNews",
        "date": "January 3, 2020",
    },
    {
        "title": "Trade Talks Continue Between Nations",
        "text": "BEIJING (Reuters) - Trade talks between the two nations continued into a second day of meetings.",
        "subject": "worldnews",
        "date": "January 4, 2020",
    },
]


@pytest.fixture
def fake_df():
    return pd.DataFrame(FAKE_ROWS)


@pytest.fixture
def real_df():
    return pd.DataFrame(REAL_ROWS)


@pytest.fixture
def two_file_dataset(tmp_path, fake_df, real_df):
    """Writes Fake.csv/True.csv into a temp dir and returns their paths."""
    fake_path = tmp_path / "Fake.csv"
    true_path = tmp_path / "True.csv"
    fake_df.to_csv(fake_path, index=False)
    real_df.to_csv(true_path, index=False)
    return fake_path, true_path


@pytest.fixture
def single_file_dataset(tmp_path, fake_df, real_df):
    """One CSV with its own 'label' column (FAKE/REAL strings), the other supported layout."""
    fake_labeled = fake_df.copy()
    real_labeled = real_df.copy()
    fake_labeled["label"] = "FAKE"
    real_labeled["label"] = "REAL"
    combined = pd.concat([fake_labeled, real_labeled], ignore_index=True)
    path = tmp_path / "dataset.csv"
    combined.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Tiny real (not mocked) TF-IDF + model artifacts, for predict.py / main.py
# tests. These are genuinely fit on a handful of synthetic sentences so the
# actual inference code path runs end-to-end -- just on toy data instead of
# the real, much larger dataset.
# --------------------------------------------------------------------------
SYNTHETIC_TEXTS = [
    # REAL-leaning (label 1)
    "senate committee approves new budget plan today",
    "city council votes on annual budget proposal",
    "central bank holds interest rates steady this week",
    "trade talks continue between two nations officials say",
    # FAKE-leaning (label 0)
    "shocking secret miracle cure doctors hate revealed today",
    "you wont believe this shocking secret trick",
    "anonymous insiders claim aliens are real secret",
    "celebrity shocking secret will blow your mind",
]
SYNTHETIC_LABELS = [1, 1, 1, 1, 0, 0, 0, 0]


@pytest.fixture
def trained_model_dir(tmp_path):
    """
    Fits tiny vectorizer/model artifacts on SYNTHETIC_TEXTS and saves them
    with the same filenames src/train.py uses, so predict.py's real loading
    and inference code is exercised without touching the project's real
    ml-service/models directory.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(SYNTHETIC_TEXTS)
    lr = LogisticRegression(max_iter=1000).fit(X, SYNTHETIC_LABELS)
    nb = MultinomialNB().fit(X, SYNTHETIC_LABELS)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(lr, model_dir / "logistic_regression.joblib")
    joblib.dump(nb, model_dir / "multinomial_nb.joblib")
    return model_dir


@pytest.fixture
def use_trained_model_dir(monkeypatch, trained_model_dir):
    """Points src.predict at trained_model_dir and makes sure its cache is clean."""
    monkeypatch.setattr(predictor, "MODEL_DIR", str(trained_model_dir))
    predictor._cache.clear()
    yield trained_model_dir
    predictor._cache.clear()
