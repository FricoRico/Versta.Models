import json
import hashlib
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any

from .dataset import load_omnidocbench, get_image_path
from .compare import calculate_character_accuracy, calculate_word_accuracy, compare_results
from .typing import EvaluationResult

# Type alias for detection results
DetectionResult = Dict[str, Any]
OCRResult = List[Dict[str, Any]]


def create_cache_key(image_path: str, model_path: str, stage: str) -> str:
    combined = f"{image_path}:{model_path}:{stage}"
    return hashlib.md5(combined.encode()).hexdigest()


def load_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache_file: Path, cache_data: dict):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)


def _crop_text_regions(
    image: np.ndarray, 
    polygons: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Crop text regions from image using detected polygons.
    
    Args:
        image: Full image array
        polygons: List of detection polygons
    
    Returns:
        List of (cropped_image, bounding_box) tuples
    """
    regions = []
    h, w = image.shape[:2]
    
    for poly in polygons:
        if len(poly) < 4:
            continue
        
        # Get bounding rectangle
        pts = poly.reshape(-1, 2)
        x, y, cw, ch = cv2.boundingRect(pts.astype(np.int32))
        
        # Clip to image boundaries
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        
        if cw <= 0 or ch <= 0:
            continue
        
        # Crop image
        crop = image[y:y+ch, x:x+cw]
        box = np.array([x, y, x + cw, y + ch], dtype=np.int32)
        regions.append((crop, box))
    
    return regions


def run_pipeline(
    image_path: str,
    detector_path: Path,
    recognizer_path: Path,
    cache_dir: Path,
    is_quantized: bool = False,
    batch_size: int = 8,
    detector_model_name: str = "PP-OCRv5_mobile_det",
    recognizer_model_name: str = "PP-OCRv5_mobile_rec",
    num_threads: int = 0,
) -> Tuple[str, float]:
    """
    Run full OCR pipeline: detection + recognition.
    
    Returns:
        Tuple of (recognized_text, confidence)
    """
    import os as _os
    cache_dir = Path(cache_dir)
    detector_cache_dir = cache_dir / "detect"
    recognizer_cache_dir = cache_dir / "recognize"
    
    detector_cache_file = detector_cache_dir / "cache.json"
    recognizer_cache_file = recognizer_cache_dir / "cache.json"
    
    detector_cache_dir.mkdir(parents=True, exist_ok=True)
    recognizer_cache_dir.mkdir(parents=True, exist_ok=True)
    
    detector_key = create_cache_key(image_path, str(detector_path), "detect")
    recognizer_key = create_cache_key(image_path, str(recognizer_path), "recognize")
    
    detector_cache = load_cache(detector_cache_file)
    recognizer_cache = load_cache(recognizer_cache_file)
    
    actual_threads = num_threads if num_threads > 0 else os.cpu_count()
    
    # Create model instances once per pipeline run
    if is_quantized:
        from .onnx_engine import ONNXDetector, ONNXRecognizer
        
        detector_model_path = detector_path
        if detector_path.is_dir():
            ort_files = list(detector_path.glob("*.ort"))
            if ort_files:
                detector_model_path = ort_files[0]
            else:
                raise FileNotFoundError(f"No .ort file found in {detector_path}")
        detector = ONNXDetector(detector_model_path, num_threads=actual_threads)
        
        recognizer_model_path = recognizer_path
        model_dir = recognizer_path
        if recognizer_path.is_dir():
            ort_files = list(recognizer_path.glob("*.ort"))
            if ort_files:
                recognizer_model_path = ort_files[0]
                model_dir = recognizer_path
            else:
                raise FileNotFoundError(f"No .ort file found in {recognizer_path}")
        recognizer = ONNXRecognizer(recognizer_model_path, model_dir, num_threads=actual_threads)
    else:
        from paddleocr import TextDetection, TextRecognition
        detector = TextDetection(
            model_name=detector_model_name,
            model_dir=None,
            device="cpu",
            cpu_threads=actual_threads,
        )
        recognizer = TextRecognition(
            model_name=recognizer_model_name,
            model_dir=None,
            device="cpu",
            cpu_threads=actual_threads,
        )
    
    # Detection step
    detect_result: Optional[DetectionResult] = None
    detect_confidence: float = 0.5
    
    if detector_key in detector_cache:
        cached = detector_cache[detector_key]
        detect_result = cached.get("result")
        detect_confidence = cached.get("confidence", 0.5)
    
    if detect_result is None:
        detect_result, detect_confidence = perform_detection(
            image_path, detector, batch_size, is_quantized
        )
        detector_cache[detector_key] = {
            "result": detect_result,
            "confidence": detect_confidence,
            "timestamp": time.time()
        }
        save_cache(detector_cache_file, detector_cache)
    
    # Recognition step - pass full image path, pipeline handles region extraction
    recognize_result: Optional[OCRResult] = None
    recognize_confidence: float = 0.5
    
    if recognizer_key in recognizer_cache:
        cached = recognizer_cache[recognizer_key]
        recognize_result = cached.get("result")
        recognize_confidence = cached.get("confidence", 0.5)
    
    if recognize_result is None:
        recognize_result, recognize_confidence = perform_recognition(
            image_path, detect_result, recognizer, batch_size, is_quantized
        )
        recognizer_cache[recognizer_key] = {
            "result": recognize_result,
            "confidence": recognize_confidence,
            "timestamp": time.time()
        }
        save_cache(recognizer_cache_file, recognizer_cache)
    
    overall_confidence = (
        (detect_confidence or 0.5 + recognize_confidence or 0.5) / 2
    )
    
    return recognize_result, overall_confidence


def perform_detection(
    image_path: str, 
    detector, 
    batch_size: int, 
    is_quantized: bool
) -> Tuple[DetectionResult, float]:
    """
    Perform text detection on image.
    
    Returns:
        Tuple of (detection_result_dict, max_confidence)
    """
    results = detector.predict(input=image_path, batch_size=batch_size)
    
    # Extract boxes and scores (convert numpy arrays to lists for JSON serialization)
    # Note: PaddleOCR 3.x returns keys at TOP LEVEL, not under 'res'
    boxes = []
    scores = []
    for result in results:
        # Handle both PaddleOCR 2.x (nested 'res') and 3.x (top-level keys)
        if isinstance(result, dict):
            # PaddleOCR 3.x format: keys at top level
            dt_polys = result.get('dt_polys', result.get('res', {}).get('dt_polys', []))
            dt_scores = result.get('dt_scores', result.get('res', {}).get('dt_scores', []))
        else:
            continue
        
        for i, poly in enumerate(dt_polys):
            score = dt_scores[i] if i < len(dt_scores) else 0.0
            # Convert numpy array to list for JSON serialization
            poly_list = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
            boxes.append(poly_list)
            scores.append(float(score))
    
    result = {
        "boxes": boxes,
        "scores": scores
    }
    
    return result, max(scores) if scores else 0.5


def perform_recognition(
    image_path: str, 
    detector_output: DetectionResult, 
    recognizer, 
    batch_size: int, 
    is_quantized: bool
) -> Tuple[OCRResult, float]:
    """
    Perform text recognition on detected regions.
    
    Args:
        image_path: Path to original image
        detector_output: Detection result containing boxes and scores
    
    Returns:
        Tuple of (list of recognition results, average confidence)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Extract text regions from detection boxes
    boxes = detector_output.get("boxes", [])
    
    if not boxes:
        return [{"res": {"rec_text": "", "rec_score": 0.0}}], 0.0
    
    # Crop each text region and run recognition
    all_results = []
    all_confidences = []
    all_results_serializable = []
    all_confidences_serializable = []
    
    if is_quantized:
        from .onnx_engine import _crop_text_regions
        
        boxes_np = [np.array(box, dtype=np.float32) for box in boxes]
        crops = _crop_text_regions(img, boxes_np)
        
        if crops:
            results = recognizer.predict(crops=crops, batch_size=len(crops))
            for result in results:
                all_results.append(result)
                rec_texts = result.get('rec_text', [])
                rec_scores = result.get('rec_score', [])
                for text, conf in zip(rec_texts, rec_scores):
                    all_results_serializable.append({
                        'rec_text': str(text),
                        'rec_score': float(conf)
                    })
                    all_confidences_serializable.append(float(conf))
    else:
        # Original PaddleOCR using temp files
        
        for box in boxes:
            if len(box) < 4:
                continue
            
            box_arr = np.array(box, dtype=np.float32)
            pts = box_arr.reshape(-1, 2).astype(np.int32)
            x, y, cw, ch = cv2.boundingRect(pts)
            
            h, w = img.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            cw = min(cw, w - x)
            ch = min(ch, h - y)
            
            if cw <= 2 or ch <= 2:
                continue
            
            crop = img[y:y+ch, x:x+cw]
            
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, crop)
            
            try:
                results = recognizer.predict(input=temp_path, batch_size=1)
                for result in results:
                    all_results.append(result)
                    conf = result.get('rec_score', result.get('res', {}).get('rec_score', 0.0))
                    all_confidences.append(conf)
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    
    # If no results, return empty
    if not all_results:
        return [{"rec_text": "", "rec_score": 0.0}], 0.0
    
    # Convert numpy floats to Python floats for JSON serialization
    # (all_results_serializable already initialized above for quantized path)
    
    for r in all_results:
        rec_text = r.get('rec_text', r.get('res', {}).get('rec_text', ''))
        rec_score = r.get('rec_score', r.get('res', {}).get('rec_score', 0.0))
        
        # Handle batch mode (ONNX with _crop_text_regions)
        if isinstance(rec_text, list):
            for text, score in zip(rec_text, rec_score):
                all_results_serializable.append({
                    'rec_text': str(text),
                    'rec_score': float(score)
                })
                all_confidences_serializable.append(float(score))
        else:
            all_results_serializable.append({
                'rec_text': str(rec_text),
                'rec_score': float(rec_score)
            })
            all_confidences_serializable.append(float(rec_score))
    
    avg_confidence = sum(all_confidences_serializable) / len(all_confidences_serializable) if all_confidences_serializable else 0.0
    
    return all_results_serializable, avg_confidence
