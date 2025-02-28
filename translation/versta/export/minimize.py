from os import remove
from pathlib import Path

def minimize(output_dir: Path):
    """
    Remove unnecessary files from the output directory.

    Args:
        output_dir (Path): Path to the output directory.
    """
    remove(output_dir / "special_tokens_map.json")
    remove(output_dir / "vocab.json")
