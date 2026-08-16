from io import StringIO
from pathlib import Path

import yaml


def write_keys(inference_yml: Path, keys_path: Path) -> int:
    """
    Extracts a recognizer's character dictionary from its `inference.yml`
    (`PostProcess.character_dict`) and writes it as one character per line.
    The runtime wraps the vocabulary with a CTC blank at index 0 and a space
    at the end, so neither is included here.

    Args:
        inference_yml (Path): The extracted model's inference.yml.
        keys_path (Path): Destination keys file.

    Returns:
        int: The number of characters written.
    """
    config = yaml.safe_load(inference_yml.read_text())
    charset = config["PostProcess"]["character_dict"]
    lines = StringIO()
    for ch in charset:
        lines.write(f"{ch}\n")
    keys_path.write_text(lines.getvalue())
    return len(charset)
