from pathlib import Path

from .convert_kokoro_to_onnx import convert_kokoro_to_onnx as convert_kokoro

def convert_model_to_onnx(model_name: str, export_dir: Path, model_format: str) -> Path:
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_name (str): Name of the pre-trained model.
        export_dir (Path): Path to the directory where the ONNX model will be saved.
        model_format (str): Type of the pre-trained model to convert (e.g., "kokoro").
    Returns:
        Path: The path to the exported ONNX model.
    """
    print(f"Exporting {model_name} to ONNX format...")

    if model_format == "kokoro":
        return convert_kokoro(model_name, export_dir)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

