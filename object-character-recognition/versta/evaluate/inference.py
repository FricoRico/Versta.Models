import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

import cv2
import numpy as np

from .onnx_engine import (
    ONNXDetector,
    ONNXRecognizer,
    _crop_text_regions,
)


@dataclass
class OCRRegion:
    """Single detected text region."""

    text: str
    confidence: float
    box: np.ndarray


@dataclass
class OCRPageResult:
    """OCR result for a single page."""

    image_path: str
    image_stem: str
    regions: List[OCRRegion]
    width: int
    height: int


def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """Load image and return with dimensions."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    return img, w, h


def sort_regions_by_reading_order(regions: List[OCRRegion]) -> List[OCRRegion]:
    """Sort detected regions by reading order (top-to-bottom, left-to-right in same row)."""
    if not regions:
        return regions

    def get_y_center(region: OCRRegion) -> float:
        box = region.box
        ys = box[:, 1]
        return (ys.min() + ys.max()) / 2

    def get_x_center(region: OCRRegion) -> float:
        box = region.box
        xs = box[:, 0]
        return (xs.min() + xs.max()) / 2

    y_centers = [get_y_center(r) for r in regions]

    sorted_indices = sorted(
        range(len(regions)), key=lambda i: (y_centers[i], get_x_center(regions[i]))
    )

    return [regions[i] for i in sorted_indices]


def run_paddleocr_with_models(
    image_path: str,
    detector,
    recognizer,
    batch_size: int = 8,
) -> OCRPageResult:
    """Run PaddleOCR pipeline on a single image using pre-built model instances."""
    img, w, h = load_image(image_path)
    image_stem = Path(image_path).stem

    det_results = detector.predict(input=image_path, batch_size=batch_size)

    boxes = []
    for result in det_results:
        if isinstance(result, dict):
            dt_polys = result.get("dt_polys", result.get("res", {}).get("dt_polys", []))
            dt_scores = result.get(
                "dt_scores", result.get("res", {}).get("dt_scores", [])
            )
        else:
            continue

        for i, poly in enumerate(dt_polys):
            score = dt_scores[i] if i < len(dt_scores) else 0.0
            poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            boxes.append((np.array(poly_list, dtype=np.float32), float(score)))

    if not boxes:
        return OCRPageResult(
            image_path=image_path,
            image_stem=image_stem,
            regions=[],
            width=w,
            height=h,
        )

    crops = _crop_text_regions(img, [b[0] for b in boxes])

    regions = []

    temp_paths = []
    for crop in crops:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)
        cv2.imwrite(temp_path, crop)
        temp_paths.append(temp_path)

    all_rec_results = []
    for i in range(0, len(temp_paths), batch_size):
        chunk = temp_paths[i : i + batch_size]
        rec_results = recognizer.predict(input=chunk, batch_size=batch_size)
        all_rec_results.extend(rec_results)

    for idx, result in enumerate(all_rec_results):
        text = result.get("rec_text", result.get("res", {}).get("rec_text", ""))
        conf = result.get("rec_score", result.get("res", {}).get("rec_score", 0.0))
        if text and idx < len(boxes):
            regions.append(
                OCRRegion(
                    text=str(text),
                    confidence=float(conf),
                    box=boxes[idx][0],
                )
            )

    for path in temp_paths:
        try:
            os.unlink(path)
        except Exception:
            pass

    regions = sort_regions_by_reading_order(regions)

    return OCRPageResult(
        image_path=image_path,
        image_stem=image_stem,
        regions=regions,
        width=w,
        height=h,
    )


def run_onnx_with_models(
    image_path: str,
    detector,
    recognizer,
    batch_size: int = 8,
) -> OCRPageResult:
    """Run ONNX pipeline on a single image using pre-built model instances."""
    img, w, h = load_image(image_path)
    image_stem = Path(image_path).stem

    det_results = detector.predict(image_path=image_path, batch_size=batch_size)

    boxes = []
    scores = []
    for result in det_results:
        dt_polys = result.get("dt_polys", [])
        dt_scores = result.get("dt_scores", [])

        for i, poly in enumerate(dt_polys):
            score = dt_scores[i] if i < len(dt_scores) else 0.0
            boxes.append(poly)
            scores.append(float(score))

    if not boxes:
        return OCRPageResult(
            image_path=image_path,
            image_stem=image_stem,
            regions=[],
            width=w,
            height=h,
        )

    crops = _crop_text_regions(img, boxes)

    if not crops:
        return OCRPageResult(
            image_path=image_path,
            image_stem=image_stem,
            regions=[],
            width=w,
            height=h,
        )

    rec_results = recognizer.predict(crops=crops, batch_size=len(crops))

    regions = []
    rec_texts = rec_results[0].get("rec_text", [])
    rec_scores = rec_results[0].get("rec_score", [])

    for i, (text, conf) in enumerate(zip(rec_texts, rec_scores)):
        if text:
            regions.append(
                OCRRegion(
                    text=str(text),
                    confidence=float(conf),
                    box=boxes[i],
                )
            )

    regions = sort_regions_by_reading_order(regions)

    return OCRPageResult(
        image_path=image_path,
        image_stem=image_stem,
        regions=regions,
        width=w,
        height=h,
    )


def ocr_result_to_markdown(result: OCRPageResult) -> str:
    """Convert OCR result to markdown text."""
    lines = [region.text for region in result.regions]
    return "\n\n".join(lines)


def save_prediction(
    result: OCRPageResult,
    output_dir: Path,
) -> Path:
    """Save OCR result as markdown file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{result.image_stem}.md"

    markdown = ocr_result_to_markdown(result)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return output_path


def run_inference(
    image_dir: Path,
    output_dir: Path,
    model_type: str = "onnx",
    detector_path: Optional[Path] = None,
    recognizer_path: Optional[Path] = None,
    model_name_det: str = "PP-OCRv5_mobile_det",
    model_name_rec: str = "PP-OCRv5_mobile_rec",
    batch_size: int = 8,
    num_threads: int = 0,
) -> List[Path]:
    """Run inference on all images in a directory."""
    from paddleocr import TextDetection, TextRecognition

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    image_files = sorted(
        [
            f
            for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    actual_threads = num_threads if num_threads > 0 else os.cpu_count()

    if model_type == "onnx":
        if not detector_path or not recognizer_path:
            raise ValueError("detector_path and recognizer_path required for ONNX mode")

        detector = ONNXDetector(detector_path, num_threads=actual_threads)
        recognizer = ONNXRecognizer(
            recognizer_path, recognizer_path, num_threads=actual_threads
        )
    else:
        detector = TextDetection(
            model_name=model_name_det,
            model_dir=None,
            device="cpu",
            cpu_threads=actual_threads,
        )
        recognizer = TextRecognition(
            model_name=model_name_rec,
            model_dir=None,
            device="cpu",
            cpu_threads=actual_threads,
        )

    output_paths = []

    for image_file in tqdm(image_files, desc="Processing images"):
        if model_type == "onnx":
            result = run_onnx_with_models(
                str(image_file),
                detector,
                recognizer,
                batch_size=batch_size,
            )
        else:
            result = run_paddleocr_with_models(
                str(image_file),
                detector,
                recognizer,
                batch_size=batch_size,
            )

        output_path = save_prediction(result, output_dir)
        output_paths.append(output_path)

    return output_paths
