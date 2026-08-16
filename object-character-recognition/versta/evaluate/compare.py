from typing import List, Dict, Any
from difflib import SequenceMatcher
import json
from pathlib import Path

# Import canonical OmniDocBench metrics (the original formula)
from versta._vendor.omnidocbench_metrics import calculate_ned

# Try to import NLTK for BLEU and METEOR
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words. Uses simple whitespace tokenization."""
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    # Fallback: simple whitespace tokenization
    return text.lower().split()


def calculate_bleu(predicted: str, ground_truth: str) -> float:
    """
    Calculate BLEU score - OmniDocBench text metric.

    BLEU measures n-gram overlap between predicted and ground truth.
    Uses smoothing to handle short sentences.

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text

    Returns:
        BLEU score (0-1 scale)
    """
    if not NLTK_AVAILABLE:
        # Return a simple ratio-based fallback
        pred_tokens = predicted.lower().split() if predicted else []
        gt_tokens = ground_truth.lower().split() if ground_truth else []
        if not pred_tokens or not gt_tokens:
            return 0.0 if pred_tokens or gt_tokens else 1.0

        # Simple n-gram overlap ratio
        common = len(set(pred_tokens) & set(gt_tokens))
        return common / max(len(pred_tokens), len(gt_tokens))

    predicted = predicted or ""
    ground_truth = ground_truth or ""

    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    try:
        # Tokenize
        ref = [ground_truth.lower().split()]
        hyp = predicted.lower().split()

        # BLEU with smoothing for short sentences
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(ref, hyp, smoothing_function=smoothing)
        return bleu
    except Exception:
        return 0.0


def calculate_meteor(predicted: str, ground_truth: str) -> float:
    """
    Calculate METEOR score - OmniDocBench text metric.

    METEOR is based on unigram precision and recall, with alignment
    to handle word ordering differences.

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text

    Returns:
        METEOR score (0-1 scale)
    """
    if not NLTK_AVAILABLE:
        # Return NED-based fallback
        return calculate_normalized_edit_distance(predicted, ground_truth)

    predicted = predicted or ""
    ground_truth = ground_truth or ""

    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    try:
        # Tokenize
        ref = _tokenize(ground_truth)
        hyp = _tokenize(predicted)

        # Calculate METEOR - requires lists
        meteor = meteor_score([ref], hyp)
        return meteor
    except Exception:
        return calculate_normalized_edit_distance(predicted, ground_truth)


def calculate_normalized_edit_distance(predicted: str, ground_truth: str) -> float:
    """
    Calculate Normalized Edit Distance (NED) - OmniDocBench's primary text metric.

    NED = 1 - (edit_distance / max(len(predicted), len(ground_truth)))

    This is the same metric OmniDocBench reports as "Edit_dist" or "Norm Edit".
    Values range from 0 (no match) to 1 (perfect match).

    Note: Now delegates to the canonical OmniDocBench formula from vendor library.

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text from dataset

    Returns:
        NED as a float between 0 and 1
    """
    # Use the canonical OmniDocBench implementation
    return calculate_ned(predicted or "", ground_truth or "")


def calculate_character_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate character-level accuracy between predicted and ground truth.

    Note: This now uses NED formula for consistency with OmniDocBench.
    Use calculate_normalized_edit_distance() for the official NED metric.

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text from dataset

    Returns:
        Character accuracy as a float between 0 and 1
    """
    # Use NED formula for character accuracy - aligns with OmniDocBench
    return calculate_normalized_edit_distance(predicted, ground_truth)


def calculate_word_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate word-level accuracy using F1-style set matching.

    This matches OmniDocBench's approach for word-level evaluation.
    Uses set-based matching (not positional) to handle different word orders.

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text from dataset

    Returns:
        Word F1 score as a float between 0 and 1
    """
    predicted = predicted or ""
    ground_truth = ground_truth or ""

    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())

    if not pred_words and not gt_words:
        return 1.0
    if not pred_words or not gt_words:
        return 0.0

    # Calculate precision, recall, F1
    true_positives = len(pred_words & gt_words)

    precision = true_positives / len(pred_words) if pred_words else 0.0
    recall = true_positives / len(gt_words) if gt_words else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_line_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate line-level accuracy using sequence matcher.

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text from dataset

    Returns:
        Line accuracy as a float between 0 and 1
    """
    predicted = predicted or ""
    ground_truth = ground_truth or ""

    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    return SequenceMatcher(None, predicted, ground_truth).ratio()


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU (Intersection over Union) between two boxes.

    Boxes are in [x, y, w, h] format or [x1, y1, x2, y2] format.
    """
    if not box1 or not box2:
        return 0.0

    # Convert to [x1, y1, x2, y2] format
    def to_xyxy(box):
        if len(box) == 4:
            x, y, w, h = box
            return [x, y, x + w, y + h]
        return box

    b1 = to_xyxy(box1)
    b2 = to_xyxy(box2)

    # Calculate intersection
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_detection_metrics(
    predicted_boxes: List[Any],
    ground_truth_boxes: List[Any],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate detection metrics (precision, recall, F1).

    Args:
        predicted_boxes: List of predicted bounding boxes
        ground_truth_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with precision, recall, F1
    """
    if not predicted_boxes and not ground_truth_boxes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}

    if not predicted_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    if not ground_truth_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    # Match predictions to ground truth
    matched_gt = set()
    ious = []

    for pred_box in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)

    # Calculate metrics
    true_positives = len(matched_gt)
    precision = true_positives / len(predicted_boxes) if predicted_boxes else 0.0
    recall = true_positives / len(ground_truth_boxes) if ground_truth_boxes else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    avg_iou = sum(ious) / len(ious) if ious else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "iou": avg_iou}


def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all evaluation results.

    Compares both PaddleOCR and quantized (ONNX) results against ground truth.
    Now includes all OmniDocBench metrics: NED, BLEU, METEOR, Word F1.

    Args:
        results: List of EvaluationResult dictionaries

    Returns:
        Dictionary with aggregate metrics for both models
    """
    if not results:
        return {
            "total_samples": 0,
            "original": {
                "avg_ned": 0.0,
                "avg_bleu": 0.0,
                "avg_meteor": 0.0,
                "avg_word_f1": 0.0,
                "avg_confidence": 0.0,
            },
            "quantized": {
                "avg_ned": 0.0,
                "avg_bleu": 0.0,
                "avg_meteor": 0.0,
                "avg_word_f1": 0.0,
                "avg_confidence": 0.0,
            },
            "comparison": {
                "ned_diff": 0.0,
                "bleu_diff": 0.0,
                "meteor_diff": 0.0,
                "word_f1_diff": 0.0,
            },
        }

    n = len(results)

    # Original model metrics (NED is now the primary metric - same as character_accuracy for backward compat)
    orig_ned = sum(r.get("character_accuracy", 0) for r in results)
    orig_bleu = sum(r.get("bleu", 0) for r in results)
    orig_meteor = sum(r.get("meteor", 0) for r in results)
    orig_word_f1 = sum(r.get("word_accuracy", 0) for r in results)
    orig_conf = sum(r.get("original_confidence", 0) for r in results)

    # Quantized model metrics
    quant_ned = sum(r.get("character_accuracy_quantized", 0) for r in results)
    quant_bleu = sum(r.get("bleu_quantized", 0) for r in results)
    quant_meteor = sum(r.get("meteor_quantized", 0) for r in results)
    quant_word_f1 = sum(r.get("word_accuracy_quantized", 0) for r in results)
    quant_conf = sum(r.get("quantized_confidence", 0) for r in results)

    return {
        "total_samples": n,
        "original": {
            "avg_ned": orig_ned / n,
            "avg_bleu": orig_bleu / n,
            "avg_meteor": orig_meteor / n,
            "avg_word_f1": orig_word_f1 / n,
            "avg_confidence": orig_conf / n,
        },
        "quantized": {
            "avg_ned": quant_ned / n,
            "avg_bleu": quant_bleu / n,
            "avg_meteor": quant_meteor / n,
            "avg_word_f1": quant_word_f1 / n,
            "avg_confidence": quant_conf / n,
        },
        "comparison": {
            "ned_diff": (orig_ned - quant_ned) / n,
            "bleu_diff": (orig_bleu - quant_bleu) / n,
            "meteor_diff": (orig_meteor - quant_meteor) / n,
            "word_f1_diff": (orig_word_f1 - quant_word_f1) / n,
            "confidence_diff": (orig_conf - quant_conf) / n,
        },
    }


def write_omnidocbench_format(
    results: List[Dict[str, Any]], output_path: Path, aggregate_metrics: Dict[str, Any]
):
    """
    Write evaluation results in OmniDocBench-compatible JSON format.

    Includes all OmniDocBench metrics: NED (Norm Edit), BLEU, METEOR, Word F1.

    Args:
        results: List of per-image results
        output_path: Output file path
        aggregate_metrics: Aggregate metrics dictionary
    """
    output_data = {
        "metadata": {
            "dataset": "OmniDocBench",
            "evaluation_type": "ocr",
            "total_samples": len(results),
            "metrics": ["NED", "BLEU", "METEOR", "Word_F1"],
        },
        "metrics": aggregate_metrics,
        "per_image": [],
    }

    for r in results:
        # NED is stored as character_accuracy for backward compatibility
        output_data["per_image"].append(
            {
                "image_path": r.get("image_path", ""),
                "ground_truth": r.get("ground_truth", ""),
                "original": {
                    "predicted_text": r.get("original_text", ""),
                    "confidence": r.get("original_confidence", 0.0),
                    "ned": r.get(
                        "character_accuracy", 0.0
                    ),  # NED = normalized edit distance
                    "bleu": r.get("bleu", 0.0),
                    "meteor": r.get("meteor", 0.0),
                    "word_f1": r.get("word_accuracy", 0.0),
                },
                "quantized": {
                    "predicted_text": r.get("quantized_text", ""),
                    "confidence": r.get("quantized_confidence", 0.0),
                    "ned": r.get("character_accuracy_quantized", 0.0),
                    "bleu": r.get("bleu_quantized", 0.0),
                    "meteor": r.get("meteor_quantized", 0.0),
                    "word_f1": r.get("word_accuracy_quantized", 0.0),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
