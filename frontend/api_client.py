"""
Thin wrapper around the Spring Boot backend's /api/v1/* endpoints.

This is the ONLY way the Streamlit app gets data: it never talks to the
Python ML service or to any model file. Every failure is raised as a
BackendError whose message is already friendly enough to show to a user.
"""
from __future__ import annotations

import os
from typing import Any

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
API_PREFIX = "/api/v1"

# (connect, read) seconds. Connecting to a local/compose service is instant, so a
# short connect timeout makes "backend is down" show up quickly; the read timeout
# leaves room for the backend's own ML call (its limit is 10s).
TIMEOUT = (3, 15)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "multinomial_nb": "Multinomial Naive Bayes",
}
DEFAULT_MODEL = "logistic_regression"


class BackendError(Exception):
    """
    A failed backend call. `kind` lets the UI pick the right presentation:
      unreachable  - backend not running / wrong URL
      timeout      - backend too slow
      invalid_input- the text was rejected (empty, too short, no known words, ...)
      ml_down      - backend is up but the ML service is down or untrained
      server       - anything else unexpected
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


_session = requests.Session()


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{BACKEND_URL}{API_PREFIX}{path}"
    try:
        response = _session.request(method, url, timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        raise BackendError(
            "unreachable",
            f"Can't reach the NewsCheck backend at {BACKEND_URL}. Make sure it is running.",
        ) from None
    except requests.exceptions.Timeout:
        raise BackendError(
            "timeout", "The backend took too long to respond. Please try again in a moment."
        ) from None
    except requests.exceptions.RequestException:
        raise BackendError("server", "Something went wrong while contacting the backend.") from None

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.ok:
        if payload is None:
            raise BackendError("server", "The backend returned a response that couldn't be read.")
        return payload

    # The backend's errors look like {"error": "...", "message": "..."}.
    message = payload.get("message") if isinstance(payload, dict) else None
    if response.status_code == 400:
        raise BackendError("invalid_input", message or "The backend rejected this request.")
    if response.status_code == 503:
        raise BackendError(
            "ml_down", message or "The analysis service is currently unavailable. Please try again shortly."
        )
    raise BackendError("server", message or f"The backend returned an error (HTTP {response.status_code}).")


def analyze(text: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """POST /analyze -> {label, probability, model, explanation, disclaimer}."""
    result = _request("POST", "/analyze", json={"text": text, "model": model})
    if not isinstance(result, dict) or "label" not in result or "probability" not in result:
        raise BackendError("server", "The backend returned an analysis in an unexpected format.")
    return result


def history(limit: int = 10) -> list[dict[str, Any]]:
    """GET /history -> recent predictions, newest first, with snake_case keys."""
    rows = _request("GET", "/history", params={"limit": limit})
    if not isinstance(rows, list):
        raise BackendError("server", "The backend returned history in an unexpected format.")
    # The backend sends text_snippet/created_at; accept camelCase too so a
    # naming-strategy change on the backend can't silently blank the table.
    return [
        {
            "text_snippet": row.get("text_snippet", row.get("textSnippet", "")),
            "model": row.get("model"),
            "label": row.get("label"),
            "probability": row.get("probability"),
            "created_at": row.get("created_at", row.get("createdAt")),
        }
        for row in rows
    ]


def metrics() -> dict[str, Any]:
    """GET /metrics -> the ML service's metrics.json object, unchanged."""
    return _request("GET", "/metrics")


def health() -> dict[str, Any]:
    """GET /health -> {status, backend, ml_service: {status, trained, models_available, message}}."""
    return _request("GET", "/health")
