from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"

PCA_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = PCA_FEATURE_COLUMNS + ["scaled_amount", "scaled_time"]

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

TRANSACTIONS_STREAM = "transactions"
TRANSACTIONS_CONSUMER_GROUP = "scorer-group"
SCORED_CHANNEL = "scored_transactions"

API_WS_URL = "ws://localhost:8000/stream"
API_BASE_URL = "http://localhost:8000"
