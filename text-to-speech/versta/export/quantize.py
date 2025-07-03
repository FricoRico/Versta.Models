from pathlib import Path

from .quantize_kokoro import quantize_kokoro
from .quantize_piper import quantize_piper

BLOCKED_NODES = []
BLOCKED_OPS = []

def quantize_model(export_dir: Path, model_filename: str, quantization_dir: Path, model_format: str):
    """
    Quantizes a specific ONNX model file and saves the quantized model to the given directory.

    Args:
        export_dir (Path): Path to the directory where the ONNX model is stored.
        model_filename (str): Name of the ONNX model file to quantize (e.g., "encoder_model.onnx").
        quantization_dir (Path): Path to the directory where the quantized model will be saved.
        model_format (str): Type of the pre-trained model to convert (e.g., "kokoro", "piper").
    """
    if model_format == "kokoro":
        return quantize_kokoro(export_dir, model_filename, quantization_dir)
    elif model_format == "piper":
        return quantize_piper(export_dir, model_filename, quantization_dir)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
