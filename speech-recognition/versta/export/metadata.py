import json

from pathlib import Path
from typing import List

from .definitions import languages as SUPPORTED_LANGUAGES
from ..version import VERSION

DEFAULT_VAD_FILENAME = "ggml-silero-v6.2.0.bin"


def generate_metadata(
        output_dir: Path,
        model_type: str,
        repo_id: str,
        model_filename: str,
        vad_filename: str,
        languages: List[str] = None,
) -> Path:
    """
    Generates the per-model metadata.json describing a whisper.cpp (ggml) model, in the
    format expected by the Versta speech-recognition module.

    Args:
        output_dir (Path): Directory where the metadata file will be written.
        model_type (str): Whisper model variant (e.g. "base-q8_0", "small.en").
        repo_id (str): Hugging Face repository id holding the whisper model.
        model_filename (str): Downloaded whisper model filename.
        vad_filename (str): Downloaded Silero-VAD model filename.
        languages (List[str]): Supported language codes. Defaults to the full Whisper
            set, or ``["en"]`` for English-only (".en") model variants.

    Returns:
        Path: The path to the written metadata file.
    """
    if languages is None:
        languages = ["en"] if model_type.endswith(".en") else list(SUPPORTED_LANGUAGES)

    metadata = {
        "id": model_type,
        "version": VERSION,
        "base_model": repo_id,
        "languages": languages,
        "architectures": ["Whisper"],
        "files": {
            "inference": {
                "model": model_filename,
                "vad": vad_filename,
            }
        },
    }

    metadata_file = output_dir / "metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    return metadata_file
