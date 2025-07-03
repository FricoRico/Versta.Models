from pathlib import Path

from .convert_kokoro_to_onnx import convert_kokoro_to_onnx as convert_kokoro
from .convert_piper_to_onnx import convert_piper_to_onnx as convert_piper

def convert_model_to_onnx(model_name: str, export_dir: Path, model_format: str, voice_path: str = None) -> Path:
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_name (str): Name of the pre-trained model.
        export_dir (Path): Path to the directory where the ONNX model will be saved.
        model_format (str): Type of the pre-trained model to convert (e.g., "kokoro", "piper").
        voice_path (str): Voice path specification for Piper models (e.g., "nl/nl_NL/mls/medium").
    Returns:
        Path: The path to the exported ONNX model.
    """
    print(f"Exporting {model_name} to ONNX format...")

    if model_format == "kokoro":
        return convert_kokoro(model_name, export_dir)
    elif model_format == "piper":
        return convert_piper(model_name, export_dir, voice_path)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

