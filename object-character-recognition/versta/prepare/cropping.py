"""Image cropping and normalization logic for calibration."""

import cv2
import numpy as np
from pathlib import Path
from typing import List

from .typing import CropResult, DetectionResult


def crop_text_regions(
    image: np.ndarray,
    detection_result: DetectionResult,
) -> List[CropResult]:
    """
    Crop detected text regions from image.

    Args:
        image: Input image (BGR format from OpenCV)
        detection_result: DetectionResult containing boxes and scores

    Returns:
        List of CropResult with cropped image, bounding box, and score
    """
    crops: List[CropResult] = []

    boxes = detection_result["boxes"]
    scores = detection_result["scores"]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crops.append(
            {
                "image": crop,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
            }
        )

    return crops


def normalize_for_inference(
    crop: np.ndarray,
    target_width: int = 320,
    target_height: int = 48,
) -> np.ndarray:
    """
    Normalize and resize crop for PaddleOCR recognizer inference.

    Preprocessing:
    1. Grayscale conversion
    2. Resize to target_size (height=48, width=320)
    3. Normalize to (-1, 1) range (divide by 255, subtract 0.5, divide by 0.5)
    4. Convert to float32
    5. Add channel dimension: [1, H, W]

    Args:
        crop: Cropped image region
        target_width: Target width (default: 320, standard PaddleOCR)
        target_height: Target height (default: 48, standard PaddleOCR)

    Returns:
        Preprocessed image with shape [1, 1, target_height, target_width]
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    scale = min(target_height / h, target_width / w, 1.0)
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_height, target_width), 0, dtype=np.uint8)
    start_h = (target_height - resized.shape[0]) // 2
    start_w = (target_width - resized.shape[1]) // 2
    padded[
        start_h : start_h + resized.shape[0],
        start_w : start_w + resized.shape[1],
    ] = resized

    normalized = (padded.astype(np.float32) / 255.0 - 0.5) / 0.5

    normalized = normalized[np.newaxis, np.newaxis, :, :]

    return normalized


def resize_for_detection(
    image: np.ndarray,
    target_width: int = 960,
    target_height: int = 960,
) -> np.ndarray:
    """
    Resize image maintaining aspect ratio for detection.

    Args:
        image: Input image (BGR format)
        target_width: Target width (default: 960)
        target_height: Target height (default: 960)

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
