import json

from pathlib import Path
from typing import List

from .typing import OCRBundleMetadata


def load_metadata(model_dir: Path) -> OCRBundleMetadata:
    """
    Loads and parses the metadata.json file from the specified folder.

    Args:
        model_dir (Path): Path to the folder containing the metadata.json file.

    Returns:
        OCRBundleMetadata: A dictionary with the directory, languages, and module.
    """
    metadata_file = model_dir / "metadata.json"

    metadata = OCRBundleMetadata(
        directory=None,
        languages=None,
        module=None
    )

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as file:
            data = json.load(file)

            metadata["directory"] = metadata_file.parent.name
            metadata["languages"] = data.get("languages", [])
            metadata["module"] = data.get("module")

        missing_entries = [key for key, value in metadata.items() if value is None]

        if missing_entries:
            raise ValueError(f"Missing required metadata entries: {', '.join(missing_entries)}")

        return metadata
    else:
        raise FileNotFoundError(f"Metadata file not found in {model_dir}")


def load_metadata_for_input_dirs(input_dirs: List[Path]) -> List[OCRBundleMetadata]:
    """
    Loads the metadata.json file from the specified input directories.

    Args:
        input_dirs (List[Path]): List of paths to the input directories containing the metadata.json files.

    Returns:
        List[OCRBundleMetadata]: List of metadata with the languages and modules.
    """
    metadata: List[OCRBundleMetadata] = list()

    for folder in input_dirs:
        metadata.append(load_metadata(folder))

    return metadata


def generate_metadata(unique_id: str, version: str, output_dir: Path, languages: List[str], modules: List[str], model_metadata: List[OCRBundleMetadata]) -> Path:
    """
    Generates a metadata file for the OCR bundle process.

    Args:
        unique_id (str): Unique identifier for the bundle.
        version (str): Version of the bundle process.
        output_dir (Path): Path to the directory where the metadata file will be saved.
        languages (List[str]): List of unique languages supported by the bundle.
        modules (List[str]): List of unique module types in the bundle.
        model_metadata (List[OCRBundleMetadata]): List of OCRBundleMetadata dictionaries containing per-model info.

    Returns:
        Path: Path to the generated metadata.json file.
    """
    metadata = {
        "id": unique_id,
        "version": version,
        "languages": languages or [],
        "modules": modules or [],
        "metadata": model_metadata or []
    }

    # Define the path for the metadata.json file
    metadata_file = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, default=serialize_metadata, indent=4)

    return metadata_file


# Custom serialization function to handle non-serializable objects (like Path)
def serialize_metadata(obj: OCRBundleMetadata) -> dict:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

