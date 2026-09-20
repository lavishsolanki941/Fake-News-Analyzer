"""
Central configuration for the ML service.

Everything here is env-var driven with sane local-dev defaults, so the
same code runs unchanged on a laptop and inside the Docker container
used by docker-compose (only the env vars differ, e.g. MODEL_DIR
pointing at a mounted volume).
"""
import os

_ML_SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory holding the raw dataset. Only ever read by the offline
# `python -m src.train` CLI script -- the live FastAPI app never
# touches this directory.
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_ML_SERVICE_ROOT, "data"))

# Directory the FastAPI app loads model.joblib / vectorizer.joblib /
# metrics.json / metadata.json from, and the directory `train` writes
# them to.
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(_ML_SERVICE_ROOT, "models"))

# --- Dataset layout ---
# Two-file layout (the common Kaggle "Fake and Real News" style):
# DATA_DIR/Fake.csv (all rows FAKE) + DATA_DIR/True.csv (all rows REAL).
FAKE_CSV_PATH = os.environ.get("FAKE_CSV_PATH", os.path.join(DATA_DIR, "Fake.csv"))
TRUE_CSV_PATH = os.environ.get("TRUE_CSV_PATH", os.path.join(DATA_DIR, "True.csv"))

# Single-file layout: one CSV that already has its own label column
# (e.g. a "label" column with FAKE/REAL or 0/1). Only used if this is
# explicitly set -- we never guess at a random CSV lying in DATA_DIR.
DATASET_PATH = os.environ.get("DATASET_PATH")

# Rows whose combined title+text is shorter than this (after cleaning)
# carry essentially no signal for TF-IDF and are dropped.
MIN_TEXT_LENGTH = int(os.environ.get("MIN_TEXT_LENGTH", "20"))

# Label convention used everywhere downstream: DB rows, API responses,
# the Streamlit UI. Keeping it in one place avoids 0/1 mix-ups later.
REAL_LABEL = 1
FAKE_LABEL = 0
LABEL_NAMES = {REAL_LABEL: "REAL", FAKE_LABEL: "FAKE"}
