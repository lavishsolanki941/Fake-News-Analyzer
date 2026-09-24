"""
Tests for src.preprocessing: stripping wire-service leakage markers,
whitespace normalization, and dropping too-short / exact-duplicate /
near-duplicate rows before training. All rows are small synthetic
DataFrames built inline -- the real dataset is never touched.
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    build_dataset,
    combine_title_text,
    normalize_whitespace,
    strip_source_artifacts,
)


# --------------------------------------------------------------------------
# Leakage stripping
# --------------------------------------------------------------------------
def test_strip_source_artifacts_removes_leading_dateline():
    text = "WASHINGTON (Reuters) - The Senate passed a bill on Tuesday."
    cleaned, was_stripped = strip_source_artifacts(text)

    assert was_stripped is True
    assert "(Reuters)" not in cleaned
    assert "WASHINGTON" not in cleaned
    assert cleaned.startswith("The Senate")


def test_strip_source_artifacts_removes_bare_agency_mention():
    text = "A spokesperson (AP) confirmed the news today."
    cleaned, was_stripped = strip_source_artifacts(text)

    assert was_stripped is True
    assert "(AP)" not in cleaned


def test_strip_source_artifacts_leaves_clean_text_untouched():
    text = "This article has no wire service markers at all."
    cleaned, was_stripped = strip_source_artifacts(text)

    assert was_stripped is False
    assert cleaned == text


# --------------------------------------------------------------------------
# Whitespace / combining
# --------------------------------------------------------------------------
def test_normalize_whitespace_collapses_and_strips():
    assert normalize_whitespace("  a   b\n\tc  ") == "a b c"


def test_combine_title_text_joins_with_period():
    df = pd.DataFrame({"title": ["Headline"], "text": ["Body text"]})
    combined = combine_title_text(df)
    assert combined.iloc[0] == "Headline. Body text"


def test_combine_title_text_strips_trailing_period_and_space():
    # combine_title_text() strips leading/trailing "." and " " from the
    # joined result, so a trailing full stop on the body is not doubled up.
    df = pd.DataFrame({"title": ["Headline"], "text": ["Body text."]})
    combined = combine_title_text(df)
    assert combined.iloc[0] == "Headline. Body text"


def test_combine_title_text_handles_missing_title_column():
    df = pd.DataFrame({"text": ["Body only"]})
    combined = combine_title_text(df)
    assert combined.iloc[0] == "Body only"


# --------------------------------------------------------------------------
# build_dataset: dropping too-short / duplicate rows, and reporting why
# --------------------------------------------------------------------------
def test_build_dataset_drops_rows_shorter_than_min_length():
    df = pd.DataFrame(
        {
            "title": ["Hi", "A Real Headline About Something"],
            "text": ["ok", "This is a full-length article body with real content in it."],
            "label": [0, 1],
        }
    )
    clean_df, report = build_dataset(df, min_text_length=20)

    assert report["dropped_too_short"] == 1
    assert len(clean_df) == 1
    assert report["rows_out"] == 1


def test_build_dataset_drops_exact_duplicates():
    df = pd.DataFrame(
        {
            "title": ["Same Headline Every Time", "Same Headline Every Time"],
            "text": [
                "Identical article body content here for testing.",
                "Identical article body content here for testing.",
            ],
            "label": [0, 0],
        }
    )
    clean_df, report = build_dataset(df, min_text_length=5)

    assert report["dropped_exact_duplicates"] == 1
    assert len(clean_df) == 1


def test_build_dataset_drops_near_duplicates_differing_only_in_punctuation():
    df = pd.DataFrame(
        {
            "title": ["Big News Today", "Big News Today!!!"],
            "text": [
                "Something happened, and it matters a lot.",
                "something happened and it matters a lot",
            ],
            "label": [1, 1],
        }
    )
    clean_df, report = build_dataset(df, min_text_length=5)

    assert report["dropped_near_duplicates"] == 1
    assert len(clean_df) == 1


def test_build_dataset_counts_leakage_artifacts_by_label():
    df = pd.DataFrame(
        {
            "title": ["Bill Passes", "Shocking Secret Revealed Today"],
            "text": [
                "WASHINGTON (Reuters) - The bill passed the Senate on a wide margin today.",
                "You will not believe this secret nobody wants you to know about at all.",
            ],
            "label": [1, 0],
        }
    )
    _, report = build_dataset(df, min_text_length=5)

    assert report["leakage_artifacts_stripped"]["REAL"] == 1
    assert report["leakage_artifacts_stripped"]["FAKE"] == 0


def test_build_dataset_keeps_distinct_rows_and_reports_class_balance():
    df = pd.DataFrame(
        {
            "title": ["First Story Here", "Second Story Here"],
            "text": [
                "This is the first distinct article with enough content.",
                "This is a second, completely different article with content.",
            ],
            "label": [0, 1],
        }
    )
    clean_df, report = build_dataset(df, min_text_length=5)

    assert len(clean_df) == 2
    assert report["class_balance"]["total_rows"] == 2
    assert report["class_balance"]["fake_count"] == 1
    assert report["class_balance"]["real_count"] == 1
