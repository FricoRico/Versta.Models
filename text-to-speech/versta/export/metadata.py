import json

from pathlib import Path
from typing import List
from os import listdir, path

from .convert_ort import ORTFiles

def generate_metadata(version: str, output_dir: Path, model: str, architectures: List[str], ort_files: ORTFiles, voices: List[str]) -> Path:
    """
    Generates a metadata file for the model conversion process.

    Args:
        version (str): Version of the model conversion process.
        model (str): Name of the model being converted.
        architectures (List[str]): List of architectures used in the model.
        output_dir (Path): Path to the directory where the metadata file will be saved.
        ort_files (ORTFiles): Dictionary containing the file paths for the encoder and decoder ORT files.
        voices (List[str]): List of voices available for the model.
    """
    metadata = {
        "version": version,
        "base_model": model,
        "architectures": architectures,
        "files": {
            "inference": ort_files or {},
            "voices": voices or []
        }
    }

    # Define the path for the metadata.json file
    metadata_file_path = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    return metadata_file_path

def get_voices(directory: Path) -> List[str]:
    """
    Get all voices from voices directory

    Args:
        directory (Path): Path to the voices directory
    """
    files = listdir(directory)

    voices = [f for f in files if path.isfile(path.join(directory, f))]

    return [path.join("voices", f) for f in voices]
