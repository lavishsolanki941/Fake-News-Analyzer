# NewsCheck ML service (FastAPI). Serves /predict, /explain, /metrics, /health.
#
# IMPORTANT: this image does NOT contain a trained model or the dataset.
# ml-service/models/ and ml-service/data/ are gitignored (see the project's
# .gitignore and .dockerignore), so nothing under them is baked into the
# image at build time. docker-compose.yml mounts ./ml-service/models from
# the host into /app/models at container start, so the container uses
# whatever you already trained locally with `python -m src.train`.
# If /app/models is empty, /predict and /explain return a clear 503
# ("model not trained yet") instead of a fabricated answer -- see
# src/predict.py's ModelNotTrainedError.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer) so `docker build` only
# re-installs packages when requirements.txt actually changes, not on
# every source edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Everything else (main.py, config.py, src/). .dockerignore keeps this from
# also pulling in data/, models/, tests/, __pycache__, etc.
COPY . .

# Matches config.py's MODEL_DIR default layout; docker-compose mounts the
# host's ./ml-service/models here (see docker-compose.yml).
ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
