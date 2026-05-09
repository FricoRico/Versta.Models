import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


def vllm_url() -> str:
    return os.environ.get("VLLM_URL", "https://127.0.0.1")


def api_key() -> str:
    return os.environ.get("VLLM_API_KEY", "******")


def default_model() -> str:
    return os.environ.get("DEFAULT_MODEL", "versta")


def request_timeout() -> int:
    return int(os.environ.get("REQUEST_TIMEOUT", "60"))


def max_retries() -> int:
    return int(os.environ.get("MAX_RETRIES", "3"))


def retry_backoff() -> int:
    return int(os.environ.get("RETRY_BACKOFF", "2"))


def sample_seed() -> int:
    return int(os.environ.get("SAMPLE_SEED", "1778419142"))


def sample_buffer() -> float:
    return float(os.environ.get("SAMPLE_BUFFER", "3.0"))
