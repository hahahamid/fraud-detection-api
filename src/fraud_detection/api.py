import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import create_model

from fraud_detection.config import (
    MODELS_DIR,
    PCA_FEATURE_COLUMNS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    SCORED_CHANNEL,
)
from fraud_detection.ensemble import combine
from fraud_detection.scorer_service import load_models, score_transaction

TransactionIn = create_model(
    "TransactionIn",
    transaction_id=(Optional[str], None),
    Time=(float, ...),
    Amount=(float, ...),
    **{column: (float, ...) for column in PCA_FEATURE_COLUMNS},
)


def get_models_dir() -> Path:
    return MODELS_DIR


def get_redis_client() -> aioredis.Redis:
    return aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = get_models_dir()
    if (models_dir / "scaler.joblib").exists():
        app.state.scaler, app.state.if_model, app.state.ae_model = load_models(models_dir)
    else:
        app.state.scaler, app.state.if_model, app.state.ae_model = None, None, None
    yield


app = FastAPI(lifespan=lifespan)


def read_metrics(models_dir: Path) -> dict:
    metrics_path = models_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file at {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/score")
def score_endpoint(transaction: TransactionIn):
    if app.state.scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    raw_data = transaction.model_dump(exclude={"transaction_id"})
    result = score_transaction(raw_data, app.state.scaler, app.state.if_model, app.state.ae_model)
    return combine(
        transaction.transaction_id or "manual",
        result["if_score"], result["if_flagged"],
        result["ae_score"], result["ae_flagged"],
    )


@app.get("/models")
def models_endpoint(models_dir: Path = Depends(get_models_dir)):
    try:
        return read_metrics(models_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket, redis_client: aioredis.Redis = Depends(get_redis_client)):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(SCORED_CHANNEL)
    await websocket.accept()
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(SCORED_CHANNEL)
        await pubsub.close()
