import json

from pathlib import Path

from ..version import VERSION


def generate_bundle_metadata(
        id: str,
        model_metadata: dict,
        output_dir: Path,
        directory: str,
) -> Path:
    """
    Generates the bundle-level metadata.json following the SpeechRecognitionBundleMetadata
    schema consumed by the Versta Android application.

    The supported languages are taken directly from the bundled model metadata, so the
    bundle automatically inherits every language the model supports.

    Args:
        id (str): Unique bundle identifier (e.g. "whisper.base-q8_0").
        model_metadata (dict): Per-model metadata produced by the export module.
        output_dir (Path): Directory where the bundle metadata file will be written.
        directory (str): Subdirectory (within the bundle) holding the model files and its
            per-model metadata, e.g. the model type (e.g. "base-q8_0").

    Returns:
        Path: The path to the written bundle metadata file.
    """
    languages = model_metadata.get("languages", [])

    bundle_metadata = {
        "id": id,
        "version": VERSION,
        "languages": languages,
        "modules": ["recognition"],
        "metadata": [
            {
                "directory": directory,
                "languages": languages,
                "module": "recognition",
            }
        ],
    }

    metadata_file = output_dir / "metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(bundle_metadata, f, indent=4)

    return metadata_file
