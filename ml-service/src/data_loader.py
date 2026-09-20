"""
Loads the raw dataset from disk and normalizes it into a single,
predictable shape: a DataFrame with at least ["title", "text", "label"]
columns, label already mapped to 0 (FAKE) / 1 (REAL).

This module does NOT clean or dedupe anything -- that's preprocessing.py.
Its only job is "find the files, read them, make sure they look sane."
Keeping that separate means a bad/missing dataset fails loudly and
early, with a message telling you exactly what to do, instead of
surfacing as a confusing error three steps later.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from config import DATA_DIR, DATASET_PATH, FAKE_CSV_PATH, FAKE_LABEL, REAL_LABEL, TRUE_CSV_PATH

REQUIRED_TEXT_COLUMNS = ("title", "text")

# Common spellings/casings we'll accept for a single-file dataset's
# label column, and the values within it. We deliberately keep this
# list short and explicit rather than trying to be clever -- if your
# dataset uses something else, rename the column/values or set
# DATASET_PATH env vars accordingly.
_LABEL_COLUMN_CANDIDATES = ("label", "class", "target")
_LABEL_VALUE_MAP = {
    "fake": FAKE_LABEL,
    "false": FAKE_LABEL,
    "0": FAKE_LABEL,
    0: FAKE_LABEL,
    "real": REAL_LABEL,
    "true": REAL_LABEL,
    "1": REAL_LABEL,
    1: REAL_LABEL,
}


class DatasetNotFoundError(Exception):
    """Raised when no usable dataset files can be found on disk."""


class DatasetFormatError(Exception):
    """Raised when dataset files exist but don't have the expected shape."""


def load_raw_dataset() -> pd.DataFrame:
    """
    Find and load the dataset using whichever layout is available.

    Precedence:
      1. DATASET_PATH env var set -> single-file mode (must have its
         own label column).
      2. Both FAKE_CSV_PATH and TRUE_CSV_PATH exist -> two-file mode.
      3. Neither -> DatasetNotFoundError telling you exactly where to
         put files.
    """
    if DATASET_PATH:
        return _load_single_file(DATASET_PATH)

    fake_exists = os.path.isfile(FAKE_CSV_PATH)
    true_exists = os.path.isfile(TRUE_CSV_PATH)

    if fake_exists and true_exists:
        return _load_two_file(FAKE_CSV_PATH, TRUE_CSV_PATH)

    if fake_exists and not true_exists:
        raise DatasetNotFoundError(
            f"Found '{FAKE_CSV_PATH}' but not '{TRUE_CSV_PATH}'.\n"
            "A binary REAL-vs-FAKE classifier needs examples of both classes.\n"
            f"Please add a True.csv (same columns: title, text, subject, date) "
            f"to: {DATA_DIR}"
        )

    if true_exists and not fake_exists:
        raise DatasetNotFoundError(
            f"Found '{TRUE_CSV_PATH}' but not '{FAKE_CSV_PATH}'.\n"
            "A binary REAL-vs-FAKE classifier needs examples of both classes.\n"
            f"Please add a Fake.csv (same columns: title, text, subject, date) "
            f"to: {DATA_DIR}"
        )

    raise DatasetNotFoundError(
        "No dataset found.\n"
        "Put your data in one of these layouts:\n"
        f"  1) Two-file layout:  {FAKE_CSV_PATH}  and  {TRUE_CSV_PATH}\n"
        f"  2) Single-file layout: set the DATASET_PATH env var to a CSV that "
        f"already has a label column ({', '.join(_LABEL_COLUMN_CANDIDATES)}) "
        f"and a 'text' and/or 'title' column.\n"
        "Then re-run training. Nothing was downloaded or invented."
    )


def dataset_source_paths() -> list[str]:
    """
    The files load_raw_dataset() reads, in a fixed order, so callers
    (e.g. the Phase 4 dataset hash) refer to exactly the same files.
    Only meaningful after load_raw_dataset() has succeeded.
    """
    if DATASET_PATH:
        return [DATASET_PATH]
    return [FAKE_CSV_PATH, TRUE_CSV_PATH]


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetFormatError(f"'{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetFormatError(f"'{path}' could not be parsed as CSV: {exc}") from exc


def _validate_text_columns(df: pd.DataFrame, path: str) -> None:
    present = [c for c in REQUIRED_TEXT_COLUMNS if c in df.columns]
    if not present:
        raise DatasetFormatError(
            f"'{path}' has columns {list(df.columns)}, but needs at least one "
            f"of {REQUIRED_TEXT_COLUMNS} to build a combined title+text field."
        )


def _load_two_file(fake_path: str, true_path: str) -> pd.DataFrame:
    fake_df = _read_csv(fake_path)
    true_df = _read_csv(true_path)
    _validate_text_columns(fake_df, fake_path)
    _validate_text_columns(true_df, true_path)

    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = FAKE_LABEL
    true_df["label"] = REAL_LABEL

    combined = pd.concat([fake_df, true_df], ignore_index=True, sort=False)
    return combined


def _load_single_file(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise DatasetNotFoundError(f"DATASET_PATH is set to '{path}' but that file doesn't exist.")

    df = _read_csv(path)
    _validate_text_columns(df, path)

    label_col = _find_label_column(df, path)
    df = df.copy()
    df["label"] = _normalize_label_column(df[label_col], path)
    return df


def _find_label_column(df: pd.DataFrame, path: str) -> str:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in _LABEL_COLUMN_CANDIDATES:
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    raise DatasetFormatError(
        f"'{path}' has columns {list(df.columns)}, but none of them look like a "
        f"label column (expected one of {_LABEL_COLUMN_CANDIDATES})."
    )


def _normalize_label_column(series: pd.Series, path: str) -> pd.Series:
    def _map_value(value) -> Optional[int]:
        if isinstance(value, str):
            value = value.strip().lower()
        return _LABEL_VALUE_MAP.get(value)

    mapped = series.map(_map_value)
    unmapped_mask = mapped.isna()
    if unmapped_mask.any():
        bad_values = sorted(set(series[unmapped_mask].astype(str)))
        raise DatasetFormatError(
            f"'{path}' has label values we don't recognize: {bad_values}. "
            f"Expected values like FAKE/REAL, true/false, or 0/1."
        )
    return mapped.astype(int)
