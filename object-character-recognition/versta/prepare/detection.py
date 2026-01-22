"""Text detection logic using PaddleOCR."""

import numpy as np
from pathlib import Path
from typing import Tuple, List
from huggingface_hub import snapshot_download
from requests import head
from paddleocr import TextDetection


def repository_exists(repo_name: str) -> bool:
    """Check if HuggingFace repository exists."""
    url = f"https://huggingface.co/{repo_name}"
    response = head(url)
    return response.status_code == 200


def load_detector(detector_source: str) -> TextDetection:
    """
    Load PaddleOCR detector from HuggingFace or local path.

    Args:
        detector_source: HuggingFace model ID or local path

    Returns:
        PaddleOCR instance configured for detection only

    Raises:
        RuntimeError: If PaddleOCR is not installed
    """
    if "/" in detector_source and repository_exists(detector_source):
        detector_model_name = detector_source.split("/")[-1]
        output = snapshot_download(repo_id=detector_source)
        detector_path = Path(output)
    else:
        detector_path = Path(detector_source)
        detector_model_name = detector_path.name

    print(f"Loading {detector_model_name} from: {detector_path.as_posix()}")

    detector = TextDetection(
        model_name=detector_model_name,
        model_dir=detector_path.as_posix(),
    )

    return detector


def detect_text_regions(
    image_path: Path,
    image: np.ndarray,
    detector: TextDetection,
    min_score: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect text regions in image using detector.

    Args:
        image_path: Path to the input image
        image: Input image (BGR format from OpenCV)
        detector: PaddleOCR detector instance
        min_score: Minimum confidence score for filtering (default: 0.5)

    Returns:
        Tuple of (boxes, scores) where:
        - boxes: numpy array of shape [N, 4] with [x1, y1, x2, y2]
        - scores: numpy array of shape [N] with confidence scores
    """
    result = detector.predict(image)

    if result is None:
        return np.array([]), np.array([])

    boxes_list = []
    scores_list = []

    # Handle new dict output format from predict()
    for res in result:
        print(f"Handling result for image: {image_path.as_posix()}")
        dt_polys = res.get("dt_polys", None)
        dt_scores = res.get("dt_scores", None)

        if dt_polys is None or dt_scores is None or len(dt_polys) == 0:
            return np.array([]), np.array([])

        # Convert 4-point polygons to bounding boxes
        for i, poly in enumerate(dt_polys):
            score = dt_scores[i] if i < len(dt_scores) else 0.0

            if score < min_score:
                continue

            # poly shape: [4, 2] -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            try:
                x_coords = poly[:, 0]
                y_coords = poly[:, 1]

                x_min = int(np.min(x_coords))
                y_min = int(np.min(y_coords))
                x_max = int(np.max(x_coords))
                y_max = int(np.max(y_coords))

                boxes_list.append([x_min, y_min, x_max, y_max])
                scores_list.append(score)
            except (ValueError, TypeError, IndexError):
                continue

    if len(boxes_list) == 0:
        return np.array([]), np.array([])

    boxes = np.array(boxes_list, dtype=np.int32)
    scores = np.array(scores_list, dtype=np.float32)

    return boxes, scores
