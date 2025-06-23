import json

from pathlib import Path

def generate_metadata(id: str, version: str, output_dir: Path, espeak_data_path: Path, open_jtalk_data_path: Path) -> Path:
    """
    Generates a metadata file for the model conversion process.

    Args:
        id (str): Unique identifier for the model conversion process.
        version (str): Version of the model conversion process.
        output_dir (Path): Path to the directory where the metadata file will be saved.
        espeak_data_path (Path): Path to the directory containing the espeak data files.
        open_jtalk_data_path (Path): Path to the directory containing the open_jtalk data files.
    """
    metadata = {
        "id": id,
        "version": version,
        "type": "tts",
        "files": {
            "espeak": espeak_data_path.name,
            "open_jtalk": open_jtalk_data_path.name,
        }
    }

    # Define the path for the metadata.json file
    metadata_file = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file
