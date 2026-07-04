# Fraud Detection API — Design

## Purpose

A learning/portfolio project that simulates real-time fraud detection, inspired by
tools like Stripe Radar. Uses the Kaggle credit card fraud dataset, two anomaly
detection models (Isolation Forest and Autoencoder), a Redis Streams-based
producer/consumer pipeline to simulate live transaction flow, a FastAPI serving
layer, and a live Streamlit dashboard.

Primary goals: learn the ML + streaming stack end-to-end, and end up with a
demoable, well-explained project for a portfolio/interviews. Not intended for
production use.

## Components

1. **Trainer** (`train.py`, offline, run once/occasionally)
   Loads `creditcard.csv`, splits train/test preserving chronological order,
   trains an Isolation Forest and an Autoencoder on legitimate transactions from
   the train split, evaluates both on the held-out test split, and saves model
   artifacts to `models/` plus a `metrics.json` report (precision/recall/F1/AUC,
   per-transaction inference latency).

2. **Producer** (`replay_producer.py`)
   Reads the held-out test-split transactions (never seen during training) and
   publishes them one at a time to a Redis Stream (`transactions`), with a
   configurable delay to simulate live traffic.

3. **Scorer** (`scorer_service.py`)
   A consumer process reading from the `transactions` Redis Stream. Loads both
   trained models, scores each incoming transaction with each model, and
   publishes the combined result to a Redis pub/sub channel (`scored_transactions`).

4. **API** (FastAPI)
   - `POST /score` — score one transaction synchronously (manual testing via curl).
   - `GET /models` — metadata/metrics for both trained models.
   - `WS /stream` — websocket that forwards scored results from the Redis
     pub/sub channel to connected clients in real time.

5. **Dashboard** (Streamlit)
   Connects to the websocket and renders a live rolling table of transactions
   (last ~200) with each model's score, color-coded flags, and a running
   flagged-vs-total count. No persistence — purely in-memory/streaming state.

## Data Flow

**Offline path** (run once before any streaming demo):
`train.py` loads the full dataset, splits chronologically (train on earlier
transactions, test on later ones, avoiding leakage of future information into
training). Both models are fit on legitimate transactions only from the train
split — they learn what "normal" looks like, then flag deviations at inference
time. Both are evaluated on the test split against known fraud labels, and each
model's flagging threshold is chosen from its precision/recall curve on that
evaluation. Results (metrics + thresholds) are saved to `models/` and
`metrics.json`.

**Online path** (the real-time demo):
`replay_producer.py` streams the test-split transactions (unseen by training)
into Redis at a configurable pace. `scorer_service.py` consumes each one, scores
it with both models, and produces a result of the form:

```json
{
  "transaction_id": "...",
  "isolation_forest": {"score": 0.87, "flagged": true},
  "autoencoder": {"score": 0.91, "flagged": true},
  "ensemble_flagged": true
}
```

`ensemble_flagged` is true if *either* model flags the transaction — simple and
explainable, and gives a concrete point of comparison ("here's where the two
models disagreed and why"). The result is published to Redis pub/sub, forwarded
by the FastAPI websocket, and rendered live in the dashboard. Nothing is
persisted beyond the in-memory rolling window in the dashboard.

## Error Handling

Kept light (this is a learning project) but present enough to demonstrate good
practice:

- Producer/Scorer: if Redis is unreachable, retry with backoff and log clearly
  rather than crashing silently.
- Scorer: a malformed transaction is logged and skipped, not allowed to kill the
  consumer loop.
- API: `/score` validates the input feature vector shape and returns a clear 422
  on mismatch; the websocket handles client disconnects gracefully.

## Testing

- Unit tests for the feature preprocessing function — must be identical between
  training and serving, to avoid train/serve skew.
- Unit tests for the ensemble flagging logic given fixed mock model outputs.
- One integration test using `fakeredis` that runs a couple of sample
  transactions through producer → scorer → output and checks the shape of the
  result.
- The offline `metrics.json` from training doubles as a sanity check that both
  models beat random guessing (AUC threshold).

## Explicit Non-Goals

- No persistence of streamed/scored transactions (dashboard state is in-memory
  only; historical analysis happens via the offline `metrics.json`, not the live
  stream).
- No Docker/containerization — runs via a local Python venv plus a locally
  installed Redis server.
- No real Kafka — Redis Streams stands in for the broker to keep setup simple
  while still exercising a real producer/consumer/broker pattern.
- Not intended for production use; no auth, rate limiting, or multi-tenancy.

## Setup Dependencies

- Python 3.11+, a virtual environment (`venv`).
- Redis server installed locally (e.g. via Homebrew) and running as a local
  service — not part of the Python app itself.
- Kaggle credit card fraud dataset (`creditcard.csv`), downloaded manually and
  placed in a `data/` directory (not committed to git, given its size and terms
  of use).
