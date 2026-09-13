import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- ClearML ---
CLEARML_PROJECT_NAME = os.getenv(
    "CLEARML_PROJECT_NAME", "media_tracking_topic_modeling"
)

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "ddreamboy")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "media-tracking-topics")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "media-tracking-data")

# --- LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://routerai.ru/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite")

# --- Embedder ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local_hf")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "2"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", "")
EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "1024"))
EMBEDDING_TORCH_DTYPE = os.getenv("EMBEDDING_TORCH_DTYPE", "float16")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "0"))
EMBEDDING_MAX_WORKERS = int(os.getenv("EMBEDDING_MAX_WORKERS", "8"))

# --- Database ---
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "")


# --- Pipeline ---
PIPELINE_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "14400"))
DRIFT_WINDOW_DAYS = int(os.getenv("DRIFT_WINDOW_DAYS", "3"))
N_FULL_CORPUS = int(os.getenv("N_FULL_CORPUS", "300000"))
RANDOM_STATE = 42

# --- Paths ---
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_HPARAMS_PATH = CONFIGS_DIR / "default_hparams.json"
THRESHOLDS_PATH = CONFIGS_DIR / "thresholds.json"

# --- BERTopic artifact stop-words ---
ARTIFACT_STOP_WORDS = [
    "truekpru",
    "truekpspbru",
    "подписаться",
    "наш",
    "известный",
    "масштабный",
    "важный",
    "подписаться truekpru",
]

# --- ClearML tags ---
TAG_PRODUCTION = "production"
TAG_ARCHIVED = "archived"
TAG_FAILED_VALIDATION = "failed_validation"
TAG_TRAINING_IN_PROGRESS = "training_in_progress"
