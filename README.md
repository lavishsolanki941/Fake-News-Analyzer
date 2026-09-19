# Fake News Classification System

> Skeleton README (Phase 1). Filled in properly in Phase 9 once every
> service exists and there are real results/metrics to describe.

## Overview
_TODO (Phase 9): what this project is, in plain terms._

## Problem Statement
_TODO: what problem this solves, and what it explicitly does NOT solve
(this is text-similarity to known real/fake examples, not fact-checking)._

## Architecture
_TODO: diagram + explanation of the three tiers and why Spring Boot
exists as a real middle tier rather than a pass-through._

```
Streamlit UI  --HTTP-->  Spring Boot API  --HTTP-->  FastAPI ML service
 (frontend/)               (backend/)                 (ml-service/)
```

## Features
_TODO_

## Tech Stack
_TODO: per-service breakdown._

## Project Structure
```
fake-news-classifier/          (this repo)
├── docker-compose.yml
├── README.md
├── .gitignore
├── ml-service/                # FastAPI + scikit-learn (all ML lives here)
├── backend/                   # Spring Boot API gateway
└── frontend/                  # Streamlit thin client
```

## Dataset Requirements
_TODO (Phase 2): expected columns, the two-file (Fake.csv/True.csv)
layout, where to place files, how path is configured._

## ML Workflow
_TODO (Phases 2-5): preprocessing, TF-IDF, training, evaluation._

## Data Leakage Caveat
_TODO: the Reuters-sourcing story — why very high accuracy on these
datasets is a red flag, not a win, and what we did about it._

## Model Comparison & Evaluation Metrics
_TODO (Phase 4): Logistic Regression vs Multinomial Naive Bayes vs
DummyClassifier baseline, with real numbers from metrics.json._

## API Contract Between Tiers
_TODO (Phase 9): full request/response shapes for all three services._

## Install / Train / Run

### Manual
_TODO_

### Docker Compose
_TODO_

## Limitations
_TODO: never claim this detects "all fake news" or verifies truth._

## Future Improvements
_TODO_
