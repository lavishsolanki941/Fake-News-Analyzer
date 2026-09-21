"""
FastAPI app for the ML service. This is the contract the Spring Boot
backend calls:

  POST /predict  {text, model} -> {label, probability, model}
  POST /explain  {text, model} -> {model, top_positive, top_negative, note}
  GET  /metrics  -> models/metrics.json
  GET  /health   -> {status, trained, models_available}

Run from the ml-service/ directory:  uvicorn main:app --reload
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import predict as predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "logistic_regression"

app = FastAPI(
    title="NewsCheck ML Service",
    description=(
        "Classifies news text as REAL or FAKE with scikit-learn models.\n\n"
        "**Important:** `probability` is the model's *confidence in its own label*, "
        "learned from patterns in a training dataset. It is NOT the probability that the "
        "story is factually true, and the model does not fact-check anything."
    ),
    version="1.0.0",
)


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    # Empty/whitespace text is deliberately allowed through validation so the
    # predictor can return its own clear "text is empty" message.
    text: str = Field(..., max_length=50_000, description="Headline or article text to analyze.")
    model: str = Field(
        DEFAULT_MODEL,
        description=f"Which model to use: {', '.join(predictor.MODEL_FILES)}.",
    )


class PredictResponse(BaseModel):
    label: Literal["REAL", "FAKE"]
    probability: float = Field(
        ...,
        description="Model confidence in `label` (0-1). NOT the probability the story is factually true.",
    )
    model: str


class WordContribution(BaseModel):
    word: str = Field(..., description="A word or two-word phrase present in the input.")
    contribution: float = Field(
        ..., description="Size of the push toward (positive) or away from (negative) the predicted label."
    )


class ExplainResponse(BaseModel):
    model: str
    top_positive: list[WordContribution] = Field(
        ..., description="Words that pushed the model toward its predicted label."
    )
    top_negative: list[WordContribution] = Field(
        ..., description="Words that pushed the model away from its predicted label."
    )
    note: str


class HealthResponse(BaseModel):
    status: str
    trained: bool
    models_available: list[str]


# --------------------------------------------------------------------------
# Error handling: clean JSON {"detail": "..."} messages, never stack traces
# --------------------------------------------------------------------------
_ERROR_STATUS = {
    predictor.ModelNotTrainedError: 503,
    predictor.ModelLoadError: 503,
    predictor.InvalidModelError: 422,
    predictor.EmptyInputError: 422,
    predictor.TextTooShortError: 422,
    predictor.NoRecognizedWordsError: 422,
}


@app.exception_handler(predictor.PredictionError)
async def prediction_error_handler(_: Request, exc: predictor.PredictionError) -> JSONResponse:
    return JSONResponse(status_code=_ERROR_STATUS.get(type(exc), 500), content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    problems = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"] if part != "body") or "request body"
        problems.append(f"{where}: {err['msg']}")
    return JSONResponse(status_code=422, content={"detail": "Invalid request. " + "; ".join(problems)})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# --------------------------------------------------------------------------
# Endpoints (plain `def` so scikit-learn work runs off the event loop)
# --------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: AnalyzeRequest) -> dict[str, Any]:
    """Classify the text as REAL or FAKE. Confidence is the model's, not a truth probability."""
    return predictor.predict(request.text, request.model)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: AnalyzeRequest) -> dict[str, Any]:
    """Show which words in the text pushed the model toward or away from its prediction."""
    return predictor.explain(request.text, request.model)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Serve the training/evaluation report (models/metrics.json)."""
    path = predictor.metrics_path()
    if not os.path.isfile(path):
        raise predictor.ModelNotTrainedError(
            "metrics.json not found. Run `python -m src.train` from the ml-service directory."
        )
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read %s", path)
        raise predictor.ModelLoadError(
            "metrics.json could not be read. Re-run `python -m src.train` to regenerate it."
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models = predictor.available_models()
    return HealthResponse(status="ok", trained=bool(models), models_available=models)
