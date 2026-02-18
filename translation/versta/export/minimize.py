from os import remove
from pathlib import Path


def minimize(output_dir: Path):
    """
    Remove unnecessary files from the output directory.

    Args:
        output_dir (Path): Path to the output directory.
    """
    files_to_remove = [
        "special_tokens_map.json",
        "vocab.json",
    ]

    for filename in files_to_remove:
        filepath = output_dir / filename
        if filepath.exists():
            remove(filepath)

    # Remove any .onnx files that shouldn't be in the final output
    for onnx_file in output_dir.glob("*.onnx"):
        onnx_file.unlink()
        print(f"Removed intermediate ONNX file: {onnx_file.name}")
