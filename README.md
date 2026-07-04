# Fraud Detection API

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009485)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E)
![Redis Streams](https://img.shields.io/badge/Redis-Streams-DC382D)

A real-time fraud detection demo inspired by tools like Stripe Radar. Trains
an Isolation Forest and an Autoencoder on the Kaggle credit card fraud
dataset, then replays held-out transactions through a Redis Streams
producer/consumer pipeline, scores them live, and displays results in a
Streamlit dashboard.

## Setup

1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install and start Redis locally (macOS via Homebrew):
   ```bash
   brew install redis
   redis-server
   ```
3. Download the Kaggle "Credit Card Fraud Detection" dataset and place it at
   `data/creditcard.csv` (the `data/` directory is gitignored).

## Running the demo

1. Train both models (one-time, or whenever retraining):
   ```bash
   python -m fraud_detection.train
   ```
   This writes `models/scaler.joblib`, `models/isolation_forest.joblib`,
   `models/autoencoder.joblib`, `models/test_holdout.csv` (the held-out
   transactions used for the live replay), and `models/metrics.json`.

2. In separate terminals, start each piece of the pipeline, in this order:
   ```bash
   # Terminal 1 (if not already running as a background service)
   redis-server

   # Terminal 2
   python -m fraud_detection.scorer_service

   # Terminal 3
   uvicorn fraud_detection.api:app --port 8000

   # Terminal 4
   streamlit run src/fraud_detection/dashboard.py

   # Terminal 5
   python -m fraud_detection.replay_producer \
     --csv-path models/test_holdout.csv --delay-seconds 0.5
   ```

3. Open the Streamlit URL printed in Terminal 4's output to watch transactions
   flow in with live fraud scores.

## Testing

```bash
pytest -v
```

All tests use `fakeredis` and synthetic fixtures — no running Redis or
downloaded dataset is required to run the test suite.

## Architecture

See `docs/superpowers/specs/2026-07-04-fraud-detection-api-design.md` for the
full design rationale.
