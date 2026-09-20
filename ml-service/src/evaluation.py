"""
Phase 4: test-set evaluation, model comparison table, and the
metrics.json report the Streamlit dashboard reads.

Everything here is computed from the actual fitted models, the actual
held-out test set and the actual dataset files -- no number in the
report is hardcoded.

metrics.json layout (schema_version 1)
--------------------------------------
{
  "schema_version": 1,
  "metadata": {
    "trained_at_utc":  ISO-8601 UTC timestamp of this training run,
    "library_versions": {"scikit-learn": str, "numpy": str, "pandas": str,
                         "python": str},
    "dataset": {
      "sha256":  hash over all source files (which data produced these numbers),
      "files":   [{"name": basename, "size_bytes": int, "sha256": str}, ...],
      "rows_in": rows read from disk before cleaning,
      "total_rows": rows after cleaning (train + test),
      "real_count": int, "fake_count": int,
      "cleaning": {"dropped_too_short": int, "dropped_exact_duplicates": int,
                   "dropped_near_duplicates": int}
    },
    "split": {"test_fraction": float, "random_state": int, "stratified": true,
              "train_size": int, "test_size": int,
              "train_class_counts": {"REAL": int, "FAKE": int},
              "test_class_counts":  {"REAL": int, "FAKE": int}},
    "leakage": {
      "wire_service_markers_stripped": {"REAL": int, "FAKE": int},
      "accuracy_warning_threshold": float,
      "models_above_threshold": [model key, ...]
    },
    "label_convention": {"REAL": 1, "FAKE": 0}
  },
  "models": {
    "<model_key>": {                       # logistic_regression, multinomial_nb, dummy_baseline
      "display_name": str,
      "is_baseline": bool,
      "test_metrics": {
        "accuracy": float,
        "macro_avg": {"precision": float, "recall": float, "f1": float},
        "per_class": {"REAL": {"precision", "recall", "f1", "support"},
                      "FAKE": {...}},
        "confusion_matrix": {
          "rows": "actual", "columns": "predicted",
          "labels": ["FAKE", "REAL"],       # index order of both axes
          "matrix": [[int, int], [int, int]],
          "cells": {"true_fake": int, "false_real": int,
                    "false_fake": int, "true_real": int}
        },
        "roc_auc": float | null,            # positive class = REAL
        "roc_auc_note": str | null
      }
    }
  },
  "comparison": {                          # derived from the numbers above
    "best_real_model_by_macro_f1": model key,
    "ranking_by_macro_f1": [model key, ...]   # best first, includes baseline
  }
}
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import FAKE_LABEL, LABEL_NAMES, REAL_LABEL

SCHEMA_VERSION = 1

# Fixed order for every per-class structure: index 0 = FAKE, 1 = REAL.
_LABEL_ORDER = [FAKE_LABEL, REAL_LABEL]
_CLASS_NAMES = [LABEL_NAMES[label] for label in _LABEL_ORDER]


# --------------------------------------------------------------------------
# Per-model metrics
# --------------------------------------------------------------------------
def _roc_auc(model, X_test, y_test) -> tuple[float | None, str | None]:
    """ROC-AUC with REAL as the positive class, or (None, reason) if unavailable."""
    if not hasattr(model, "predict_proba"):
        return None, "model has no predict_proba"

    classes = list(model.classes_)
    if REAL_LABEL not in classes:
        return None, "model never saw the REAL class"

    scores = model.predict_proba(X_test)[:, classes.index(REAL_LABEL)]
    auc = float(roc_auc_score(y_test, scores))

    note = None
    if np.unique(scores).size == 1:
        note = (
            "model outputs the same probability for every article, so it "
            "cannot rank REAL above FAKE; 0.5 means no better than chance"
        )
    return auc, note


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    """All test-set metrics for one fitted model (test data only)."""
    y_pred = model.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=_LABEL_ORDER, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=_LABEL_ORDER, average="macro", zero_division=0
    )

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(_CLASS_NAMES)
    }

    # Rows = actual class, columns = predicted class, both in _LABEL_ORDER.
    cm = confusion_matrix(y_test, y_pred, labels=_LABEL_ORDER)
    fake_i, real_i = _LABEL_ORDER.index(FAKE_LABEL), _LABEL_ORDER.index(REAL_LABEL)

    roc_auc, roc_note = _roc_auc(model, X_test, y_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_avg": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "rows": "actual",
            "columns": "predicted",
            "labels": _CLASS_NAMES,
            "matrix": cm.tolist(),
            "cells": {
                "true_fake": int(cm[fake_i, fake_i]),
                "false_real": int(cm[fake_i, real_i]),  # FAKE article called REAL
                "false_fake": int(cm[real_i, fake_i]),  # REAL article called FAKE
                "true_real": int(cm[real_i, real_i]),
            },
        },
        "roc_auc": roc_auc,
        "roc_auc_note": roc_note,
    }


def evaluate_models(
    models: Mapping[str, Any],
    display_names: Mapping[str, str],
    baseline_keys: Sequence[str],
    X_test,
    y_test,
) -> dict[str, dict[str, Any]]:
    """Evaluate every model on the same test set; keyed like `models`."""
    return {
        key: {
            "display_name": display_names[key],
            "is_baseline": key in baseline_keys,
            "test_metrics": evaluate_model(model, X_test, y_test),
        }
        for key, model in models.items()
    }


# --------------------------------------------------------------------------
# Dataset provenance
# --------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dataset_files(paths: Sequence[str]) -> dict[str, Any]:
    """
    Hash the raw dataset files so a metrics.json can be tied to the exact
    data that produced it. The combined hash covers each file's name and
    content hash in the order given, so swapping/renaming files changes it.
    """
    files = []
    combined = hashlib.sha256()
    for path in paths:
        name = os.path.basename(path)
        file_hash = _sha256_file(path)
        files.append({"name": name, "size_bytes": os.path.getsize(path), "sha256": file_hash})
        combined.update(f"{name}:{file_hash}\n".encode("utf-8"))
    return {"sha256": combined.hexdigest(), "files": files}


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------
def _class_counts(y) -> dict[str, int]:
    counts = pd.Series(y).value_counts()
    return {LABEL_NAMES[label]: int(counts.get(label, 0)) for label in (REAL_LABEL, FAKE_LABEL)}


def build_metrics_report(
    *,
    model_results: dict[str, dict[str, Any]],
    preprocessing_report: Mapping[str, Any],
    dataset_hash: Mapping[str, Any],
    y_train,
    y_test,
    test_fraction: float,
    random_state: int,
    leakage_threshold: float,
) -> dict[str, Any]:
    """Assemble the full metrics.json structure (see module docstring)."""
    balance = preprocessing_report["class_balance"]
    stripped = preprocessing_report["leakage_artifacts_stripped"]

    above_threshold = [
        key
        for key, result in model_results.items()
        if not result["is_baseline"] and result["test_metrics"]["accuracy"] > leakage_threshold
    ]

    ranking = sorted(
        model_results,
        key=lambda key: model_results[key]["test_metrics"]["macro_avg"]["f1"],
        reverse=True,
    )
    real_ranking = [key for key in ranking if not model_results[key]["is_baseline"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "library_versions": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "python": platform.python_version(),
            },
            "dataset": {
                **dataset_hash,
                "rows_in": int(preprocessing_report["rows_in"]),
                "total_rows": int(balance["total_rows"]),
                "real_count": int(balance["real_count"]),
                "fake_count": int(balance["fake_count"]),
                "cleaning": {
                    "dropped_too_short": int(preprocessing_report["dropped_too_short"]),
                    "dropped_exact_duplicates": int(preprocessing_report["dropped_exact_duplicates"]),
                    "dropped_near_duplicates": int(preprocessing_report["dropped_near_duplicates"]),
                },
            },
            "split": {
                "test_fraction": test_fraction,
                "random_state": random_state,
                "stratified": True,
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "train_class_counts": _class_counts(y_train),
                "test_class_counts": _class_counts(y_test),
            },
            "leakage": {
                "wire_service_markers_stripped": {
                    "REAL": int(stripped["REAL"]),
                    "FAKE": int(stripped["FAKE"]),
                },
                "accuracy_warning_threshold": leakage_threshold,
                "models_above_threshold": above_threshold,
            },
            "label_convention": {LABEL_NAMES[REAL_LABEL]: REAL_LABEL, LABEL_NAMES[FAKE_LABEL]: FAKE_LABEL},
        },
        "models": model_results,
        "comparison": {
            "best_real_model_by_macro_f1": real_ranking[0] if real_ranking else None,
            "ranking_by_macro_f1": ranking,
        },
    }


def save_metrics(report: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")


# --------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------
def _fmt_auc(auc: float | None) -> str:
    return f"{auc:.4f}" if auc is not None else "n/a"


def format_comparison_tables(model_results: Mapping[str, Mapping[str, Any]]) -> str:
    """Plain-ASCII comparison tables (overall, per-class, confusion matrices)."""
    lines: list[str] = []
    name_w = max(len(r["display_name"]) for r in model_results.values())

    header = f"{'Model':<{name_w}}  {'Accuracy':>8}  {'Macro P':>8}  {'Macro R':>8}  {'Macro F1':>8}  {'ROC-AUC':>8}"
    lines += ["OVERALL (test set)", header, "-" * len(header)]
    for r in model_results.values():
        m = r["test_metrics"]
        lines.append(
            f"{r['display_name']:<{name_w}}  {m['accuracy']:>8.4f}  {m['macro_avg']['precision']:>8.4f}  "
            f"{m['macro_avg']['recall']:>8.4f}  {m['macro_avg']['f1']:>8.4f}  {_fmt_auc(m['roc_auc']):>8}"
        )

    header = f"{'Model':<{name_w}}  {'Class':<5}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Support':>7}"
    lines += ["", "PER CLASS (test set)", header, "-" * len(header)]
    for r in model_results.values():
        for class_name in _CLASS_NAMES[::-1]:  # REAL first, then FAKE
            c = r["test_metrics"]["per_class"][class_name]
            lines.append(
                f"{r['display_name']:<{name_w}}  {class_name:<5}  {c['precision']:>9.4f}  "
                f"{c['recall']:>7.4f}  {c['f1']:>7.4f}  {c['support']:>7}"
            )

    lines += ["", "CONFUSION MATRICES (rows = actual, columns = predicted)"]
    for r in model_results.values():
        cm = r["test_metrics"]["confusion_matrix"]
        lines += [
            f"{r['display_name']}",
            f"  {'':<12}{'pred FAKE':>10}{'pred REAL':>10}",
            f"  {'actual FAKE':<12}{cm['matrix'][0][0]:>10}{cm['matrix'][0][1]:>10}",
            f"  {'actual REAL':<12}{cm['matrix'][1][0]:>10}{cm['matrix'][1][1]:>10}",
        ]
        if r["test_metrics"]["roc_auc_note"]:
            lines.append(f"  note: ROC-AUC {r['test_metrics']['roc_auc_note']}")

    return "\n".join(lines)
