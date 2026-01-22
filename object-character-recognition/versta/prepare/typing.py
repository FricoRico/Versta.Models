"""Type definitions for calibration data preparation module."""

from typing import TypedDict, List
import numpy as np


class DetectionResult(TypedDict):
    """Result of text detection on an image."""

    boxes: np.ndarray
    scores: np.ndarray
    image_path: str


class CropResult(TypedDict):
    """Result of cropping a text region."""

    image: np.ndarray
    box: List[int]
    score: float


class CalibrationFiles(TypedDict):
    """Paths to calibration data files."""

    tensors: str
    manifest: str


class ManifestEntry(TypedDict):
    """Entry in calibration manifest."""

    image_path: str
    box: List[int]
    score: float
    crop_index: int
    language_code: str


class LanguageCalibrationInfo(TypedDict):
    """Calibration information per language."""

    lang_code: str
    num_samples: int
    avg_confidence: float
    min_samples_reached: bool
