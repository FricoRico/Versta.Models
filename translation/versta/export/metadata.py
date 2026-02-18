import json

from pathlib import Path
from typing import List, Dict, Any

from .tokenizer import TokenizerFiles
from .convert_ort import ORTFiles
from .config import ArchitectureConfig


def generate_metadata(
    version: str,
    output_dir: Path,
    model: str,
    source_language: str,
    target_language: str,
    architectures: List[str],
    arch_config: ArchitectureConfig,
    score: float,
    tokenizer_files: TokenizerFiles,
    ort_files: ORTFiles,
) -> str:
    """
    Generates a metadata file for the model conversion process.

    Args:
        version (str): Version of the model conversion process.
        output_dir (Path): Path to the directory where the metadata file will be saved.
        model (str): Name of the model to be converted.
        source_language (str): Source language for the translation model.
        target_language (str): Target language for the translation model.
        architectures (List[str]): List of architectures used in the model.
        arch_config (ArchitectureConfig): Architecture configuration for decoder cache initialization.
        score (float): Score of the model to be converted.
        tokenizer_files (TokenizerFiles): Dictionary containing the file paths for the tokenizer files.
        ort_files (ORTFiles): Dictionary containing the file paths for the encoder and decoder ORT files.
    """
    metadata = {
        "version": version,
        "base_model": model,
        "source_language": source_language,
        "target_language": target_language,
        "architectures": architectures,
        "architecture_config": arch_config,
        "score": score,
        "files": {"tokenizer": tokenizer_files or {}, "inference": ort_files or {}},
    }

    # Define the path for the metadata.json file
    metadata_file = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file
