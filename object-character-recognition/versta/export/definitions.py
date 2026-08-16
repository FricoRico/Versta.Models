from typing import List

from .typing import FoldVariants, ModelSpec

# Folded det variants: folding the final DBNet deconv (ConvTranspose.2) is
# exact (2x2 stride-2 deconv -> 1x1 conv with spatially averaged weights) and
# preserves box scores; it is the variant shipped for live mode. The quarter
# variant additionally folds ConvTranspose.0 through an intervening BN+ReLU,
# which lowers box scores by ~0.11 — kept available for stills-only use.
FOLD_VARIANTS: FoldVariants = {
    "half": ["ConvTranspose.2"],
    "quarter": ["ConvTranspose.2", "ConvTranspose.0"],
}

QUARTER_VARIANT_NOTE = (
    "stills-only: folding ConvTranspose.0 lowers box scores by ~0.11 on the tiny tier"
)

PACK_NAME = "paddle-ocr-v6"

MODELS: List[ModelSpec] = [
    {
        "stem": "PP-OCRv6_tiny_det_infer",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_infer.tar",
        "kind": "detector",
        "opset": 14,
        "pir": True,
        "tier": "tiny",
        "script": "",
    },
    {
        "stem": "PP-OCRv6_tiny_rec_infer",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_infer.tar",
        "kind": "recognizer",
        "opset": 14,
        "pir": True,
        "tier": "tiny",
        "script": "latin",
    },
    {
        "stem": "PP-OCRv6_small_rec_infer",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_infer.tar",
        "kind": "recognizer",
        "opset": 14,
        "pir": True,
        "tier": "small",
        "script": "cj",
    },
    {
        "stem": "language_classification_infer",
        "url": "https://paddleclas.bj.bcebos.com/models/PULC/inference/language_classification_infer.tar",
        "kind": "scriptClassifier",
        "opset": 11,
        "pir": False,
        "tier": "",
        "script": "",
    },
    {
        "stem": "PP-LCNet_x0_25_textline_ori_infer",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar",
        "kind": "textlineOrientation",
        "opset": 14,
        "pir": True,
        "tier": "",
        "script": "",
    },
]


def mnn_filename(spec: ModelSpec, variant: str = "") -> str:
    """
    Resolves the output MNN file name for a model spec, matching the names the
    app's OCR catalog expects.

    Args:
        spec (ModelSpec): The model catalog entry.
        variant (str): Folded detector variant ("half"/"quarter") or empty.

    Returns:
        str: The output file name, e.g. "PP-OCRv6_tiny_det_half_int8.mnn".

    Raises:
        ValueError: If the model kind has no naming rule.
    """
    kind = spec["kind"]
    if kind == "detector":
        suffix = f"_{variant}" if variant else ""
        return f"PP-OCRv6_{spec['tier']}_det{suffix}_int8.mnn"
    if kind == "recognizer":
        return f"PP-OCRv6_{spec['tier']}_rec_int8.mnn"
    if kind == "scriptClassifier":
        return "PULC_int8.mnn"
    if kind == "textlineOrientation":
        return "textline_ori_x0_25_wq8.mnn"
    raise ValueError(f"Unknown model kind: {kind}")


def keys_filename(spec: ModelSpec) -> str:
    """
    Resolves the keys (character dictionary) file name for a recognizer tier.
    Each tier ships its own charset — the tiny tier drops Japanese kana.

    Args:
        spec (ModelSpec): A recognizer model catalog entry.

    Returns:
        str: The keys file name, e.g. "PP-OCRv6_tiny_keys.txt".
    """
    if spec["kind"] != "recognizer":
        raise ValueError("Keys files only exist for recognizer models")
    return f"PP-OCRv6_{spec['tier']}_keys.txt"
