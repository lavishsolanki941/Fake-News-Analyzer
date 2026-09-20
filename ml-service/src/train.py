"""
Phase 3: TF-IDF vectorization + train/test split + baseline vs. real
models, all trained and scored on the SAME split so their numbers are
directly comparable.

Deliberately excludes precision/recall/F1/confusion matrix/ROC-AUC and
the model comparison report -- that's Phase 4 (src/evaluation.py).
This script's only job is: vectorize, split, train, persist, and print
a quick accuracy sanity check.

Run from the ml-service/ directory with: python -m src.train
"""
from __future__ import annotations

import logging
import os
import sys

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from config import MODEL_DIR, TEST_SIZE
from src.data_loader import DatasetFormatError, DatasetNotFoundError, load_raw_dataset
from src.preprocessing import build_dataset

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Test accuracy above this on a REAL model (not the Dummy baseline) is
# treated as a leakage red flag for this dataset, not a result to
# celebrate -- see the README's data leakage caveat.
LEAKAGE_ACCURACY_THRESHOLD = 0.98

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
LOGISTIC_REGRESSION_PATH = os.path.join(MODEL_DIR, "logistic_regression.joblib")
MULTINOMIAL_NB_PATH = os.path.join(MODEL_DIR, "multinomial_nb.joblib")


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        # Unigrams alone lose word order; bigrams pick up short phrases
        # ("breaking news", "fake news") a bag-of-single-words can't see.
        ngram_range=(1, 2),
        # Ignore terms in fewer than 5 documents -- typos/rare tokens
        # that can't generalize and would just add noise + memory.
        min_df=5,
        # Ignore terms in more than 90% of documents -- too common to be
        # discriminative even after English stopwords are removed.
        max_df=0.9,
        # Cap the vocabulary at the 50k highest-scoring terms so model
        # size/memory stay bounded and rare high-cardinality terms can't
        # dominate.
        max_features=50000,
        # Use 1 + log(tf) instead of raw term frequency so one very
        # repetitive document can't swamp a feature's weight.
        sublinear_tf=True,
        # Drop common English function words (the, is, and, ...) that
        # carry no real-vs-fake signal.
        stop_words="english",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        raw_df = load_raw_dataset()
    except (DatasetNotFoundError, DatasetFormatError) as exc:
        print(f"Training aborted: {exc}", file=sys.stderr)
        sys.exit(1)

    clean_df, _report = build_dataset(raw_df)

    X = clean_df["combined_text"]
    y = clean_df["label"]

    # Split BEFORE fitting TF-IDF. Fitting first would let the
    # vectorizer's vocabulary/IDF weights be learned from the very
    # documents we later test on -- leaking test-set information into
    # training and making test accuracy look better than it really is.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)  # fit on TRAIN ONLY
    X_test = vectorizer.transform(X_test_text)  # test text is only ever transformed, never fit

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Dummy (most_frequent baseline)": DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE
        ),
    }

    print(f"Train rows: {X_train.shape[0]}  |  Test rows: {X_test.shape[0]}  |  Vocabulary size: {len(vectorizer.vocabulary_)}\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: train accuracy = {train_acc:.4f}, test accuracy = {test_acc:.4f}")

        is_baseline = name.startswith("Dummy")
        if not is_baseline and test_acc > LEAKAGE_ACCURACY_THRESHOLD:
            print(
                f"  WARNING: test accuracy {test_acc:.4f} is above the "
                f"{LEAKAGE_ACCURACY_THRESHOLD:.0%} leakage threshold. This dataset is known to "
                "carry wire-service source patterns -- Phase 4 evaluation should look closely "
                "before trusting this number.",
                file=sys.stderr,
            )

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(models["Logistic Regression"], LOGISTIC_REGRESSION_PATH)
    joblib.dump(models["Multinomial Naive Bayes"], MULTINOMIAL_NB_PATH)

    print(f"\nSaved vectorizer            -> {VECTORIZER_PATH}")
    print(f"Saved Logistic Regression   -> {LOGISTIC_REGRESSION_PATH}")
    print(f"Saved Multinomial NB        -> {MULTINOMIAL_NB_PATH}")


if __name__ == "__main__":
    main()
