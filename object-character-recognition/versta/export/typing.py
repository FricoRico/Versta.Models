from typing import TypedDict, Optional


class ModelFiles(TypedDict):
    """TFLite model files."""

    model: str
    quantization: Optional[str]


class TokenizerFiles(TypedDict):
    vocabulary: str
