from pathlib import Path

import torch
from dotenv import load_dotenv

# Load .env for HF_TOKEN (auto-picked by HuggingFace SDK)
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def get_dtype() -> torch.dtype:
    """
    Returns the best available floating point dtype for training.

    Prefers bfloat16 if supported by the current GPU, falls back to float16.

    Returns:
        torch.bfloat16 or torch.float16
    """
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
