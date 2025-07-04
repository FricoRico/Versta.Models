from shutil import rmtree, copy2
from pathlib import Path

from .convert_piper_to_onnx import get_language_from_config

def copy_folder(src: Path, dest: Path):
    """
    Copies the contents of the source directory to the destination directory.

    Args:
        src (Path): Source directory path.
        dest (Path): Destination directory
    """
    if src.exists() and src.is_dir():
        dest = dest / src.name
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                copy_folder(item, dest)
            else:
                copy2(item, dest_item)

def remove_folder(dir: Path):
    """
    Removes the specified directory and all its contents.
    """
    if dir.exists() and dir.is_dir():
        rmtree(dir)

def output_folder(output_dir: Path, model: str, model_format: str, voice: str) -> Path:
    """
    Updates output directory for specific models based on language information from config.

    Args:
        output_dir (Path): Current output directory
        model (str): Name of the model being converted
        model_format (str): Format of the model ("kokoro" or "piper")
        voice (str): Voice path specification for Piper models (e.g., "nl/nl_NL/mls/medium")

    Returns:
        Path: Updated output directory with Piper language naming
    """
    if model_format != "piper":
        return output_dir

    language_info = get_language_from_config(model, voice)
    if language_info and "family" in language_info:
        language_family = language_info["family"]

        if "output" in str(output_dir) and f"piper-{language_family}" not in str(output_dir):
            parent_dir = output_dir.parent
            output_dir = parent_dir / f"piper-{language_family}"
            print(f"Updated output directory to: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir