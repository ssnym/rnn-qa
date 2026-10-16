import os
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _str(name: str, default: str) -> str:
    # `or default` also covers variables set to an empty value
    return os.getenv(name) or default


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _path(name: str, default: str) -> Path:
    path = Path(_str(name, default))
    return path if path.is_absolute() else ROOT_DIR / path


# Runtime
MODEL_VERSION = _str("MODEL_VERSION", "1.0.0")
DEVICE = torch.device(_str("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

# Paths
MODEL_PATH = _path("MODEL_PATH", "artifact/rnn_checkpoint.pt")
DATA_FILE = _path("DATA_FILE", "data/qa-text-data.csv")

# Model architecture
EMBEDDING_DIM = _int("EMBEDDING_DIM", 50)
HIDDEN_DIM = _int("HIDDEN_DIM", 64)

# Tokenizer
UNK_TOKEN = _str("UNK_TOKEN", "<UNK>")
STRIP_CHARS = _str("STRIP_CHARS", "?'")

# Inference
CONFIDENCE_THRESHOLD = _float("CONFIDENCE_THRESHOLD", 0.40)
FALLBACK_ANSWER = _str("FALLBACK_ANSWER", "I don't Know")
DEFAULT_QUESTION = _str("DEFAULT_QUESTION", "What is the capital of France?")

# Training
EPOCHS = _int("EPOCHS", 20)
LEARNING_RATE = _float("LEARNING_RATE", 0.001)