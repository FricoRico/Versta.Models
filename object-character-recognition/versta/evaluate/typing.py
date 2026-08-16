from typing import TypedDict, List, Optional, Any
from pathlib import Path


class EvaluationResult(TypedDict):
    image_path: str
    ground_truth: str
    original_text: str
    quantized_text: str
    original_confidence: float
    quantized_confidence: float
    character_accuracy: float  # Original model accuracy
    character_accuracy_quantized: float
    word_accuracy: float  # Original model accuracy
    word_accuracy_quantized: float
    ned_original: float  # Normalized Edit Distance (lower is better)
    ned_quantized: float


class EvaluationMetrics(TypedDict):
    """Aggregate evaluation metrics."""

    total_samples: int
    avg_character_accuracy_original: float
    avg_character_accuracy_quantized: float
    avg_word_accuracy_original: float
    avg_word_accuracy_quantized: float
    avg_original_confidence: float
    avg_quantized_confidence: float
    avg_ned_original: float  # NED - lower is better
    avg_ned_quantized: float


class CacheEntry(TypedDict):
    result: str
    confidence: Optional[float]
    timestamp: float


class OCRSample(TypedDict):
    """Single OCR sample result."""

    image_path: str
    predicted_text: str
    ground_truth: str
    confidence: float
    boxes: List[Any]
    scores: List[float]
