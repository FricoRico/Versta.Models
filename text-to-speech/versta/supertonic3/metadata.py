"""
Writer for the bundle-level `metadata.json` (the app-load contract).

The shape is pinned by the app-side parser `VoiceModelMetadata` /
`VoiceInferenceFilesMetadata` (verbalized on NEU-68 via the NEU-70
workstream): a four-key `models` map plus the `tts.json` engine config,
the unicode indexer as tokenizer, and the ten voice-style files. All paths
are relative to the model root. App deserializers run
`ignoreUnknownKeys = true`, but this writer intentionally emits exactly the
contract keys and nothing else.
"""

import json

from pathlib import Path
from typing import Dict

from .sources import PASSTHROUGH_FILES, VOICE_STYLE_FILES
from .typing import MODEL_STAGES

BASE_MODEL = "Supertone/supertonic-3"
ARCHITECTURES = ["Supertonic3"]

TTS_CONFIG = PASSTHROUGH_FILES["onnx/tts.json"]  # "tts.json"
UNICODE_INDEXER = PASSTHROUGH_FILES["onnx/unicode_indexer.json"]


def build_metadata(version: str) -> Dict:
    """
    Builds the metadata.json payload for the bundle.

    Args:
        version (str): Bundle version, e.g. "v2.0.0" (from versta/version.txt).

    Returns:
        dict: Contract-shaped metadata payload.
    """
    return {
        "version": version,
        "base_model": BASE_MODEL,
        "architectures": ARCHITECTURES,
        "files": {
            "inference": {
                "models": {stage: f"{stage}.mnn" for stage in MODEL_STAGES},
                "config": TTS_CONFIG,
            },
            "tokenizer": {"unicode_indexer": UNICODE_INDEXER},
            "voices": [
                VOICE_STYLE_FILES[f"voice_styles/{name}.json"]
                for name in ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
            ],
        },
    }


def write_metadata(output_dir: Path, version: str) -> Path:
    """
    Writes metadata.json into the bundle staging directory.

    Args:
        output_dir (Path): Bundle staging root (the "model root" the app's
            relative paths resolve against).
        version (str): Bundle version.

    Returns:
        Path: The written metadata.json path.
    """
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(build_metadata(version), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return metadata_file
