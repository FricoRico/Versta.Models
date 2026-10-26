from pathlib import Path

from .convert_piper_to_onnx import convert_piper_to_onnx as convert_piper

KOKORO_ARCHIVE_NOTICE = (
    "Kokoro export is archived (versta/export/_archive/convert_kokoro_to_onnx.py); "
    "TTS ships the Supertonic 3 MNN bundle (versta.supertonic3)."
)


def convert_model_to_onnx(
    model_name: str, export_dir: Path, model_format: str, voice: str
) -> Path:
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_name (str): Name of the pre-trained model.
        export_dir (Path): Path to the directory where the ONNX model will be saved.
        model_format (str): Type of the pre-trained model to convert ("piper").
        voice (str): Voice path specification for Piper models (e.g., "nl/nl_NL/mls/medium").
    Returns:
        Path: The path to the exported ONNX model.
    Raises:
        ValueError: For any unsupported format, Kokoro included (archived).
    """
    print(f"Exporting {model_name} to ONNX format...")

    if model_format == "piper":
        return convert_piper(model_name, export_dir, voice)
    elif model_format == "kokoro":
        raise ValueError(KOKORO_ARCHIVE_NOTICE)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
