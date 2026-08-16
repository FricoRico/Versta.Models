import json

from pathlib import Path
from typing import List

from .typing import BundleMetadata


def load_metadata(model_dir: Path) -> BundleMetadata:
    """
    Loads and parses the metadata.json file from the specified folder.

    Args:
        folder (Path): Path to the folder containing the metadata.json file.

    Returns:
        dict: A dictionary with the source and target languages.
    """
    metadata_file = model_dir / "metadata.json"

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as file:
            data = json.load(file)

            metadata = BundleMetadata(
                directory=Path(metadata_file.parent.name),
                source_language=data.get("source_language"),
                target_language=data.get("target_language"),
                architecture=data.get("architecture"),
                base_model=data.get("base_model"),
                score=data.get("score"),
                version=data.get("version"),
                files=data.get("files"),
            )

        missing_entries = [key for key, value in metadata.items() if value is None]

        if missing_entries:
            raise ValueError(
                f"Missing required metadata entries: {', '.join(missing_entries)}"
            )

        return metadata
    else:
        raise FileNotFoundError(f"BundleMetadata file not found in {model_dir}")


def load_metadata_for_input_dirs(input_dirs: List[Path]) -> List[BundleMetadata]:
    """
    Loads the metadata.json file from the specified input directories.

    Args:
        input_dirs (List[Path]): List of paths to the input directories containing the metadata.json files.

    Returns:
        List[BundleMetadata]: List of metadata with the source and target languages.
    """
    metadata: List[BundleMetadata] = list()

    for folder in input_dirs:
        metadata.append(load_metadata(folder))

    return metadata


def generate_metadata(
    version: str,
    output_dir: Path,
    languages: List[str],
    language_metadata: List[BundleMetadata],
    bidirectional: bool,
) -> Path:
    """
    Generates a metadata file for the model conversion process.

    Args:
        version (str): Version of the model conversion process.
        output_dir (Path): Path to the directory where the metadata file will be saved.
        languages (List[str]): List of languages supported by the model.
        language_metadata (List[BundleMetadata]): List of BundleMetadata dictionaries containing source and target language pairs.
        bidirectional (bool): Flag to indicate if the metadata contains bidirectional language pairs.
    """
    metadata = {
        "version": version,
        "languages": languages or [],
        "bidirectional": bidirectional,
        "metadata": language_metadata or [],
    }

    # Define the path for the metadata.json file
    metadata_file = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, default=serialize_metadata, indent=4)

    return metadata_file


# Custom serialization function to handle non-serializable objects (like Path)
def serialize_metadata(obj: BundleMetadata) -> str:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
