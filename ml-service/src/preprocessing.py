"""
Cleans a raw dataset (as returned by data_loader.load_raw_dataset) into
something safe to train on: combined title+text, source-leakage
artifacts stripped and logged, duplicates removed, empty/near-empty
rows dropped.

Everything here runs BEFORE the train/test split (Phase 3). That
ordering matters: if we deduped or fit anything AFTER splitting, the
same or near-identical article could leak between train and test,
making test accuracy look better than it really is.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from config import MIN_TEXT_LENGTH

logger = logging.getLogger(__name__)

# Real articles in this dataset are almost all Reuters wire copy and
# start with a dateline like "WASHINGTON (Reuters) - ...". A model can
# hit suspiciously high accuracy by just detecting "(Reuters)" instead
# of learning anything about the writing itself -- that's the data
# leakage this project is built to call out rather than hide.
_LEADING_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9.,'\- ]{0,80}\(Reuters\)\s*-\s*", re.IGNORECASE)
_BARE_AGENCY_RE = re.compile(r"\((?:Reuters|AP|Associated Press)\)", re.IGNORECASE)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _column_or_blank(df: pd.DataFrame, column: str) -> pd.Series:
    """Return df[column] as strings, or a same-length blank Series if absent."""
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def combine_title_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate title + text into one field; either may be missing."""
    title = _column_or_blank(df, "title")
    text = _column_or_blank(df, "text")
    return (title + ". " + text).str.strip(". ").str.strip()


def strip_source_artifacts(text: str) -> tuple[str, bool]:
    """
    Remove wire-service leakage markers from one piece of text.

    Returns (cleaned_text, was_stripped) so callers can tally how often
    this happened -- that count is the evidence for the leakage warning
    in the training report, not just a claim.
    """
    stripped = False

    new_text, n = _LEADING_DATELINE_RE.subn("", text)
    if n:
        stripped = True
        text = new_text

    new_text, n = _BARE_AGENCY_RE.subn("", text)
    if n:
        stripped = True
        text = new_text

    return text, stripped


def normalize_whitespace(text: str) -> str:
    """Light normalization only: collapse whitespace, no stemming/case changes."""
    text = text.replace("\xa0", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _near_dup_key(text: str) -> str:
    """
    Normalized key used to catch "obvious" near-duplicates: the same
    article with different punctuation/casing/spacing. This is NOT
    fuzzy/semantic matching (that's out of scope here) -- it only
    catches near-identical formatting variants of the same text.
    """
    key = text.lower()
    key = _PUNCT_RE.sub("", key)
    return _WHITESPACE_RE.sub(" ", key).strip()


def report_class_balance(df: pd.DataFrame) -> dict[str, Any]:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "total_rows": total,
        "fake_count": int(counts.get(0, 0)),
        "real_count": int(counts.get(1, 0)),
        "fake_pct": round(100 * counts.get(0, 0) / total, 2) if total else 0.0,
        "real_pct": round(100 * counts.get(1, 0) / total, 2) if total else 0.0,
    }


def build_dataset(df: pd.DataFrame, min_text_length: int = MIN_TEXT_LENGTH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the full cleaning pipeline and return (clean_df, report).

    clean_df has columns: combined_text, label.
    report is a JSON-serializable dict of what happened at each step,
    meant to be folded into training metadata so the numbers in the
    README/metrics are traceable, not asserted.
    """
    report: dict[str, Any] = {"rows_in": len(df)}

    working = df.copy()

    # Strip leakage artifacts from the article body BEFORE combining with
    # the title. The dateline pattern is anchored to the start of the
    # body text -- if we combined first, the title would always sit in
    # front of it and the anchor would never match.
    text_series = _column_or_blank(working, "text")
    title_series = _column_or_blank(working, "title")

    stripped_flags = text_series.apply(strip_source_artifacts)
    cleaned_text = stripped_flags.apply(lambda pair: pair[0])
    was_stripped = stripped_flags.apply(lambda pair: pair[1])

    working["combined_text"] = (title_series + ". " + cleaned_text).str.strip(". ").str.strip()

    leakage_by_label = (
        working.assign(_stripped=was_stripped)
        .groupby("label")["_stripped"]
        .sum()
        .to_dict()
    )
    report["leakage_artifacts_stripped"] = {
        "REAL": int(leakage_by_label.get(1, 0)),
        "FAKE": int(leakage_by_label.get(0, 0)),
    }
    if report["leakage_artifacts_stripped"]["REAL"] or report["leakage_artifacts_stripped"]["FAKE"]:
        logger.warning(
            "Stripped wire-service source artifacts (e.g. '(Reuters)' datelines) from "
            "%d REAL and %d FAKE rows. This is expected to be lopsided toward REAL -- "
            "that lopsidedness IS the data leakage this pipeline guards against.",
            report["leakage_artifacts_stripped"]["REAL"],
            report["leakage_artifacts_stripped"]["FAKE"],
        )

    working["combined_text"] = working["combined_text"].apply(normalize_whitespace)

    before = len(working)
    working = working[working["combined_text"].str.len() >= min_text_length]
    report["dropped_too_short"] = before - len(working)

    before = len(working)
    working = working.drop_duplicates(subset=["combined_text"], keep="first")
    report["dropped_exact_duplicates"] = before - len(working)

    before = len(working)
    working["_near_dup_key"] = working["combined_text"].apply(_near_dup_key)
    working = working.drop_duplicates(subset=["_near_dup_key"], keep="first")
    working = working.drop(columns=["_near_dup_key"])
    report["dropped_near_duplicates"] = before - len(working)

    working = working[["combined_text", "label"]].reset_index(drop=True)
    report["rows_out"] = len(working)
    report["class_balance"] = report_class_balance(working)

    return working, report


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data_loader import DatasetFormatError, DatasetNotFoundError, load_raw_dataset

    try:
        raw_df = load_raw_dataset()
        clean_df, run_report = build_dataset(raw_df)
    except (DatasetNotFoundError, DatasetFormatError) as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(run_report, indent=2))
    print(f"\nSample cleaned row:\n{clean_df.iloc[0]['combined_text'][:200]}...")
