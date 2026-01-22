"""Metadata generation for calibration data."""

import json
import numpy as np
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Dict

from .typing import CalibrationFiles


class MetadataSerializer(json.JSONEncoder):
    """Custom JSON encoder for metadata serialization."""

    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__dict__"):
            return vars(o)
        return f"<{o.__class__.__name__}>"


def generate_calibration_metadata(
    version: str,
    output_dir: Path,
    detector_model: str,
    target_model: str,
    module: str,
    num_samples: int,
    calibration_files: CalibrationFiles,
    tensor_info: Dict | None = None,
    language_stats: Dict[str, Dict] | None = None,
) -> None:
    """
    Generate metadata.json for calibration data.

    Args:
        version: Module version
        output_dir: Output directory path
        detector_model: HuggingFace repo or path of detector model
        target_model: HuggingFace repo or path of target model
        module: Module type ("detector" or "recognizer")
        num_samples: Number of calibration samples
        calibration_files: Paths to calibration data files
        tensor_info: Dictionary with tensor shape and preprocessing info
        language_stats: Dict mapping language code to stats dict
    """
    print(f"\n{'=' * 70}")
    print(f"Generating calibration metadata")
    print(f"{'=' * 70}")

    metadata = {
        "format": "onnx_calibration_tensors",
        "version": version,
        "detector_model": detector_model,
        "target_model": target_model,
        "module": module,
        "samples": num_samples,
        "created_at": dt.now(timezone.utc).isoformat(),
        "files": calibration_files,
    }

    if tensor_info is not None:
        metadata["tensor_info"] = tensor_info

    if language_stats is not None:
        metadata["language_stats"] = language_stats

    metadata_file = output_dir / "metadata.json"

    print(f"Saving metadata to: {metadata_file}")

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, cls=MetadataSerializer)

    print(f"✓ Metadata saved")
    print(f"{'=' * 70}\n")


def serialize_metadata(obj):
    """JSON serializer for Path objects (legacy)."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")
