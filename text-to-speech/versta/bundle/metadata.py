import json

from pathlib import Path
from typing import List, TypedDict

class BundleMetadata(TypedDict):
    directory: Path

def load_metadata(model_dir: Path) -> BundleMetadata:
    """
    Loads and parses the metadata.json file from the specified folder.

    Args:
        model_dir (Path): Path to the folder containing the metadata.json file.

    Returns:
        dict: A dictionary with the source and target languages.
    """
    metadata_file = model_dir / "metadata.json"

    metadata = BundleMetadata(
        directory=None,
    )

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata["directory"] = metadata_file.parent.name

        missing_entries = [key for key, value in metadata.items() if value is None]

        if missing_entries:
            raise ValueError(f"Missing required metadata entries: {', '.join(missing_entries)}")

        return metadata
    else:
        raise FileNotFoundError(f"BundleMetadata file not found in {model_dir}")

def load_metadata_for_input_dirs(input_dir: Path) -> BundleMetadata:
    """
    Loads the metadata.json file from the specified input directories.

    Args:
        input_dir (Path): Path to the directory containing the metadata.json files.

    Returns:
       BundleMetadata: List of metadata with the source and target languages.
    """
    metadata: BundleMetadata = load_metadata(input_dir)

    return metadata

def generate_metadata(version: str, output_dir: Path, metadata: BundleMetadata) -> Path:
    """
    Generates a metadata file for the model conversion process.

    Args:
        version (str): Version of the model conversion process.
        output_dir (Path): Path to the directory where the metadata file will be saved.
        metadata (BundleMetadata): List of BundleMetadata dictionaries containing source and target language pairs.
    """
    metadata = {
        "version": version,
        "metadata": metadata
    }

    # Define the path for the metadata.json file
    metadata_file_path = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, default=serialize_metadata, indent=4)

    return metadata_file_path

# Custom serialization function to handle non-serializable objects (like Path)
def serialize_metadata(obj: BundleMetadata) -> dict:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")