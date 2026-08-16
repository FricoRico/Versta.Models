import json
import math
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

try:
    import pyclipper

    _PYCLIPPER_AVAILABLE = True
except ImportError:
    _PYCLIPPER_AVAILABLE = False


def _resolve_model_file(path: Path) -> Path:
    """Resolve a model path to an actual .ort file."""
    if path.is_dir():
        ort_files = list(path.glob("*.ort"))
        if not ort_files:
            raise FileNotFoundError(f"No .ort model file found in {path}")
        return ort_files[0]
    return path


def _load_metadata(model_dir: Path) -> Dict[str, Any]:
    """Load metadata.json from model directory."""
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


# ============================================================================
# DETECTION PREPROCESSING - Match PaddleX's DetResizeForTest
# ============================================================================


def _preprocess_image(
    img: np.ndarray,
    limit_side_len: int = 960,
    limit_type: str = "max",
    max_side_limit: int = 4000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess image for detection model - matches PaddleX's DetResizeForTest.

    Key fixes from original:
    1. Uses RGB color space (model trained on RGB via ReadImage(format="RGB"))
    2. Resizes to multiples of 32, not center-padded square
    3. Uses limit_side_len with limit_type="max" (PaddleOCR defaults)
    """
    h, w = img.shape[:2]

    # Convert BGR to RGB (PaddleX uses ReadImage(format="RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PaddleX DetResizeForTest.resize_image_type0 logic
    if limit_type == "max":
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0
    elif limit_type == "min":
        if min(h, w) < limit_side_len:
            if h < w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0
    else:
        raise ValueError(f"Unknown limit_type: {limit_type}")

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    # Apply max_side_limit if needed
    if max(resize_h, resize_w) > max_side_limit:
        ratio = float(max_side_limit) / max(resize_h, resize_w)
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

    # Round to multiples of 32 (PaddleX requirement)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    # Resize with aspect ratio preserved
    img_resized = cv2.resize(img, (resize_w, resize_h))

    # Normalize: /255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img_data = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std

    # CHW format
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.ascontiguousarray(img_data, dtype=np.float32)

    # Scale factors for coordinate transformation
    scale_factors = {
        "scale_w": w / resize_w,
        "scale_h": h / resize_h,
        "orig_h": h,
        "orig_w": w,
        "ratio_h": resize_h / h,
        "ratio_w": resize_w / w,
        "resized_h": resize_h,
        "resized_w": resize_w,
    }

    return img_data, scale_factors


# ============================================================================
# DETECTION POSTPROCESSING - Match PaddleX's DBPostProcess
# ============================================================================


def _get_mini_boxes(contour: np.ndarray) -> Tuple[List[List[float]], float]:
    """Get minimum area rotated rectangle boxes - matches PaddleX."""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """Calculate box score using mean value in box region - matches PaddleX."""
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))

    if xmax <= xmin or ymax <= ymin:
        return 0.0

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _unclip_box(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """Unclip box using pyclipper - matches PaddleX."""
    if not _PYCLIPPER_AVAILABLE:
        center = box.mean(axis=0)
        expanded = box.copy()
        for i in range(len(box)):
            vec = box[i] - center
            expanded[i] = box[i] + vec * (unclip_ratio - 1.0)
        return expanded

    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    distance = area * unclip_ratio / length

    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    try:
        expanded = np.array(offset.Execute(distance))
    except ValueError:
        expanded = np.array(offset.Execute(distance)[0])

    return expanded


def _postprocess_detections(
    map_data: np.ndarray,
    scale_factors: Dict[str, Any],
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    unclip_ratio: float = 1.5,
    max_candidates: int = 1000,
    min_size: int = 3,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Postprocess detection model output - matches PaddleX's DBPostProcess.boxes_from_bitmap.

    Key fixes from original:
    1. Uses rotated rectangles from minAreaRect (not axis-aligned boundingRect)
    2. Applies proper box_score_fast on probability map
    3. Uses pyclipper for unclip expansion
    4. Uses PaddleOCR defaults: thresh=0.3, box_thresh=0.6, unclip_ratio=2.0
    """
    if map_data.ndim == 4:
        map_data = map_data[0]
    if map_data.ndim == 3:
        map_data = map_data[0]

    pred = map_data > thresh
    bitmap = pred.astype(np.uint8)

    height, width = bitmap.shape
    dest_width = scale_factors["orig_w"]
    dest_height = scale_factors["orig_h"]
    width_scale = dest_width / width
    height_scale = dest_height / height

    outs = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        contours = outs[1]
    else:
        contours, _ = outs[0], outs[1]

    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []

    for index in range(num_contours):
        contour = contours[index]
        points, sside = _get_mini_boxes(contour)

        if sside < min_size:
            continue

        points = np.array(points)

        score = _box_score_fast(map_data, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        box = _unclip_box(points, unclip_ratio).reshape(-1, 1, 2)
        box, sside = _get_mini_boxes(box)

        if sside < min_size + 2:
            continue

        box = np.array(box)

        for i in range(box.shape[0]):
            box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
            box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

        boxes.append(box.astype(np.float32))
        scores.append(score)

    return boxes, scores


# ============================================================================
# TEXT REGION CROPPING - Match PaddleX's CropByPolys
# ============================================================================


def _get_minarea_rect_crop(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop text region using minimum area rectangle - matches PaddleX."""
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_d = 0, 1
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0

    index_b, index_c = 2, 3
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    return _get_rotate_crop_image(img, np.array(box))


def _get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop and rotate text region using perspective transform - matches PaddleX."""
    assert len(points) == 4, "shape of points must be 4x2"

    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )

    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )

    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )

    dst_img_height, dst_img_width = dst_img.shape[:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img


def _crop_text_regions(img: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
    """Crop text regions from image using detected polygons."""
    crops = []
    h, w = img.shape[:2]

    for poly in boxes:
        if len(poly) < 4:
            continue

        pts = poly.reshape(-1, 2).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        try:
            crop = _get_minarea_rect_crop(img, pts)
            crops.append(crop)
        except Exception:
            continue

    return crops


# ============================================================================
# RECOGNITION PREPROCESSING - Match PaddleX's OCRReisizeNormImg
# ============================================================================


def _preprocess_for_recognition(
    img: np.ndarray, target_height: int = 48, max_width: int = 320
) -> Tuple[np.ndarray, float]:
    """
    Preprocess image for recognition model - matches PaddleX's OCRReisizeNormImg.

    Key fixes from original:
    1. Uses RGB color space (model trained on RGB)
    2. Pads with zeros (0.0 after normalization), not 0.5
    3. max_width=320 matches rec_image_shape from PaddleX config
    """
    h, w = img.shape[:2]
    imgC = img.shape[2] if len(img.shape) == 3 else 1

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate width based on aspect ratio
    wh_ratio = w * 1.0 / h if h > 0 else 1.0
    resized_w = int(target_height * wh_ratio)

    if resized_w > max_width:
        resized_w = max_width

    # Resize
    resized_image = cv2.resize(img, (resized_w, target_height))

    # Normalize: /255, -0.5, /0.5 (matches PaddleX)
    resized_image = resized_image.astype(np.float32)
    resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5

    # Pad with zeros (0.0 after normalization - matches PaddleX)
    # Original incorrectly used 0.5
    img_data = np.zeros((imgC, target_height, max_width), dtype=np.float32)
    img_data[:, :, 0:resized_w] = resized_image

    img_data = np.expand_dims(img_data, axis=0)

    return img_data, resized_w / w


def _preprocess_for_recognition_batch(
    imgs: List[np.ndarray], target_height: int = 48, max_width: int = 320
) -> Tuple[np.ndarray, List[int]]:
    """Preprocess multiple images for recognition in batch."""
    if not imgs:
        return np.array([]), []

    processed = []
    width_scales = []

    for img in imgs:
        img_prep, scale = _preprocess_for_recognition(img, target_height, max_width)
        processed.append(img_prep[0])
        width_scales.append(scale)

    max_w = max(img.shape[2] for img in processed)

    batch = []
    for img in processed:
        if img.shape[2] < max_w:
            pad = np.zeros(
                (img.shape[0], img.shape[1], max_w - img.shape[2]), dtype=np.float32
            )
            img = np.concatenate([img, pad], axis=2)
        batch.append(img)

    batch = np.expand_dims(batch, axis=0)
    batch = np.ascontiguousarray(batch, dtype=np.float32)

    return batch, width_scales


# ============================================================================
# CTC DECODING
# ============================================================================


def _decode_ctc(logits: np.ndarray, vocab: Dict[int, str]) -> Tuple[str, float]:
    """Decode CTC greedy (argmax) output - matches PaddleX's CTCLabelDecode."""
    if logits.ndim == 3:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = np.argmax(logits, axis=-1)

    if preds.ndim > 1:
        preds = preds[0]

    if logits.ndim == 3:
        confs_per_pos = np.max(logits[0], axis=-1)
    else:
        confs_per_pos = np.ones(preds.shape)

    decoded = []
    decoded_confs = []
    prev_idx = -1
    for pos, idx in enumerate(preds):
        if idx != prev_idx and idx != 0:
            decoded.append(int(idx))
            decoded_confs.append(confs_per_pos[pos])
        prev_idx = idx

    text = ""
    for idx in decoded:
        char = vocab.get(idx, "")
        if char:
            text += char

    conf = np.mean(decoded_confs) if decoded_confs else 0.0

    return text, conf


# ============================================================================
# ONNX DETECTOR
# ============================================================================


class ONNXDetector:
    def __init__(self, model_path: Path, device: str = "cpu", num_threads: int = 0):
        self.model_path = _resolve_model_file(Path(model_path))
        self.device = device

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if num_threads > 0:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        model_dir = self.model_path.parent
        self._config = _load_metadata(model_dir)

    def predict(
        self,
        input: str = None,
        batch_size: int = 1,
        image_path: str = None,
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """Run detection inference."""
        img_path = input if input else image_path
        if not img_path:
            raise ValueError("Image path required")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img_data, scale_factors = _preprocess_image(
            img, limit_side_len=limit_side_len, limit_type=limit_type
        )

        outputs = self.session.run(None, {self.input_name: img_data})
        map_data = outputs[0]

        boxes, scores = _postprocess_detections(
            map_data,
            scale_factors,
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )

        return [{"dt_polys": boxes, "dt_scores": scores}]


class ONNXRecognizer:
    def __init__(self, model_path: Path, vocab_path: Path, num_threads: int = 0):
        self.model_path = _resolve_model_file(Path(model_path))
        self.vocab_path = Path(vocab_path)

        self.vocab = self._load_vocab(vocab_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if num_threads > 0:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        model_dir = self.model_path.parent
        self._config = _load_metadata(model_dir)

    def _load_vocab(self, vocab_path: Path) -> Dict[int, str]:
        """Load vocabulary from YAML config or binary file."""
        binary_vocab = None
        if os.path.isdir(vocab_path):
            binary_vocab = os.path.join(vocab_path, "vocab.bin")

        if not os.path.isfile(binary_vocab):
            raise FileNotFoundError(f"Vocabulary file not found: {binary_vocab}")

        vocab = {}
        with open(binary_vocab, "rb") as f:
            while True:
                chars = b""
                while True:
                    b = f.read(1)
                    if not b or b == b"\x00":
                        break
                    chars += b
                if not chars:
                    break
                char = chars.decode("utf-8", errors="replace")
                idx_bytes = f.read(4)
                if len(idx_bytes) < 4:
                    break
                idx = int.from_bytes(idx_bytes, "little")
                vocab[idx + 1] = char

        vocab[len(vocab) + 1] = " "

        return vocab

    def predict(
        self,
        input: str = None,
        batch_size: int = 1,
        image_path: str = None,
        crops: List[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Run recognition inference."""
        if crops is not None and len(crops) > 0:
            imgs = crops
        else:
            img_path = input if input else image_path
            if not img_path:
                raise ValueError("Image path required")

            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")

            imgs = [img]

        all_texts = []
        all_confs = []

        for img in imgs:
            img_data, _ = _preprocess_for_recognition(img)

            outputs = self.session.run(None, {self.input_name: img_data})
            logits = outputs[0]

            text, conf = _decode_ctc(logits, self.vocab)
            all_texts.append(text)
            all_confs.append(conf)

        return [{"rec_text": all_texts, "rec_score": all_confs}]
