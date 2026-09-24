"""
Tests for src.data_loader: finding and reading the dataset in both
supported layouts (two-file Fake.csv/True.csv, and a single labeled CSV),
and failing loudly with a clear message -- not a silent wrong answer --
when files are missing or malformed. All data is a handful of hand-written
synthetic rows from conftest.py; the real ml-service/data/*.csv is never
read by these tests.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import data_loader


# --------------------------------------------------------------------------
# Two-file layout
# --------------------------------------------------------------------------
def test_two_file_layout_loads_and_labels_correctly(monkeypatch, two_file_dataset):
    fake_path, true_path = two_file_dataset
    monkeypatch.setattr(data_loader, "DATASET_PATH", None)
    monkeypatch.setattr(data_loader, "FAKE_CSV_PATH", str(fake_path))
    monkeypatch.setattr(data_loader, "TRUE_CSV_PATH", str(true_path))

    df = data_loader.load_raw_dataset()

    assert len(df) == 8
    assert set(df["label"].unique()) == {0, 1}
    assert (df.loc[df["label"] == 0, "title"] == "You Won't Believe This Shocking Secret").any()
    assert (df.loc[df["label"] == 1, "title"] == "Senate Passes Budget Bill").any()


def test_missing_both_files_raises_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "DATASET_PATH", None)
    monkeypatch.setattr(data_loader, "FAKE_CSV_PATH", str(tmp_path / "Fake.csv"))
    monkeypatch.setattr(data_loader, "TRUE_CSV_PATH", str(tmp_path / "True.csv"))

    with pytest.raises(data_loader.DatasetNotFoundError):
        data_loader.load_raw_dataset()


def test_only_fake_csv_present_raises_not_found_naming_the_missing_file(monkeypatch, tmp_path, fake_df):
    fake_path = tmp_path / "Fake.csv"
    fake_df.to_csv(fake_path, index=False)
    monkeypatch.setattr(data_loader, "DATASET_PATH", None)
    monkeypatch.setattr(data_loader, "FAKE_CSV_PATH", str(fake_path))
    monkeypatch.setattr(data_loader, "TRUE_CSV_PATH", str(tmp_path / "True.csv"))

    with pytest.raises(data_loader.DatasetNotFoundError, match="True.csv"):
        data_loader.load_raw_dataset()


def test_two_file_layout_requires_title_or_text_column(monkeypatch, tmp_path, real_df):
    bad_fake = pd.DataFrame({"headline": ["oops"], "subject": ["News"], "date": ["2020"]})
    fake_path = tmp_path / "Fake.csv"
    true_path = tmp_path / "True.csv"
    bad_fake.to_csv(fake_path, index=False)
    real_df.to_csv(true_path, index=False)
    monkeypatch.setattr(data_loader, "DATASET_PATH", None)
    monkeypatch.setattr(data_loader, "FAKE_CSV_PATH", str(fake_path))
    monkeypatch.setattr(data_loader, "TRUE_CSV_PATH", str(true_path))

    with pytest.raises(data_loader.DatasetFormatError):
        data_loader.load_raw_dataset()


# --------------------------------------------------------------------------
# Single labeled-CSV layout
# --------------------------------------------------------------------------
def test_single_file_layout_maps_string_labels(monkeypatch, single_file_dataset):
    monkeypatch.setattr(data_loader, "DATASET_PATH", str(single_file_dataset))

    df = data_loader.load_raw_dataset()

    assert len(df) == 8
    assert set(df["label"].unique()) == {0, 1}


def test_single_file_layout_accepts_numeric_and_true_false_labels(monkeypatch, tmp_path, fake_df, real_df):
    fake_labeled = fake_df.copy()
    real_labeled = real_df.copy()
    fake_labeled["label"] = 0
    real_labeled["label"] = "true"  # mixed style on purpose: 0/1 alongside true/false spellings
    path = tmp_path / "numeric.csv"
    pd.concat([fake_labeled, real_labeled], ignore_index=True).to_csv(path, index=False)
    monkeypatch.setattr(data_loader, "DATASET_PATH", str(path))

    df = data_loader.load_raw_dataset()

    assert set(df["label"].unique()) == {0, 1}


def test_single_file_missing_path_raises_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "DATASET_PATH", str(tmp_path / "does_not_exist.csv"))

    with pytest.raises(data_loader.DatasetNotFoundError):
        data_loader.load_raw_dataset()


def test_single_file_missing_label_column_raises_format_error(monkeypatch, tmp_path):
    df = pd.DataFrame({"title": ["a"], "text": ["b"]})
    path = tmp_path / "nolabel.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(data_loader, "DATASET_PATH", str(path))

    with pytest.raises(data_loader.DatasetFormatError):
        data_loader.load_raw_dataset()


def test_single_file_unrecognized_label_values_raise_format_error(monkeypatch, tmp_path):
    df = pd.DataFrame({"title": ["a", "b"], "text": ["c", "d"], "label": ["maybe", "REAL"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(data_loader, "DATASET_PATH", str(path))

    with pytest.raises(data_loader.DatasetFormatError, match="maybe"):
        data_loader.load_raw_dataset()


def test_single_file_empty_csv_raises_format_error(monkeypatch, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    monkeypatch.setattr(data_loader, "DATASET_PATH", str(path))

    with pytest.raises(data_loader.DatasetFormatError):
        data_loader.load_raw_dataset()
