import json

from pathlib import Path
from typing import List
from os import listdir, path

from .utils import copy_folder
from .typing import ORTFiles, TokenizerFiles


def generate_metadata(version: str, output_dir: Path, model: str, model_format: str, ort_files: ORTFiles,
                      tokenizer_files: TokenizerFiles, voices: List[str]) -> Path:
    """
    Generates a metadata file for the model conversion process.

    Args:
        version (str): Version of the model conversion process.
        model (str): Name of the model being converted.
        model_format (str): Format of the model ("kokoro" or "piper").
        output_dir (Path): Path to the directory where the metadata file will be saved.
        ort_files (ORTFiles): Dictionary containing the file paths for the encoder and decoder ORT files.
        tokenizer_files (TokenizerFiles): Dictionary containing the file paths for the tokenizer files.
        voices (List[str]): List of voices available for the model.
    """
    architectures = _get_model_architectures(model_format)

    metadata = {
        "version": version,
        "type": "voice",
        "base_model": model,
        "architectures": architectures,
        "files": {
            "inference": ort_files or {},
            "tokenizer": tokenizer_files or {},
            "voices": voices or []
        }
    }

    # Define the path for the metadata.json file
    metadata_file = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file


def get_voices(input_path: Path, export_path: Path, model_format: str = "kokoro") -> List[str]:
    """
    Get all voices from voices directory

    Args:
        input_path (Path): Path to the voices directory
        export_path (Path): Path to the directory where the voices will be exported
        model_format (str): Format of the model ("kokoro" or "piper")
    """
    if model_format == "kokoro":
        copy_folder(input_path, export_path)

        files = listdir(export_path)
        voices = [f for f in files if path.isfile(path.join(export_path, f))]
        return [path.join("voices", f) for f in voices]
    elif model_format == "piper":
        return []
    else:
        raise ValueError(f"Unsupported model format: {model_format}")


def _get_model_architectures(model_format: str) -> List[str]:
    """
    Get the architectures used in the model.

    Args:
        model_format (str): Format of the model ("kokoro" or "piper").

    Returns:
        List[str]: List of architectures used in the model.
    """
    if model_format == "kokoro":
        return ["StyleTTS2"]
    elif model_format == "piper":
        return ["VITS"]
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
