import json
from pathlib import Path
from struct import pack


def convert_vocabulary(export_dir: Path, config_file: Path) -> Path | None:
    """
     Extracts the vocabulary from the Piper model config.json and saves it in the export directory.

     Args:
         export_dir (Path): Path to the directory where the vocabulary will be saved.
         config_file (Path): Path to the configuration file of the Piper model.

     Returns:
         Path: Path to the created vocab.bin file.
     """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    source_vocabulary = _load_vocabulary(config_file)

    if not source_vocabulary:
        return None

    optimized_vocabulary = export_dir / "vocab.bin"

    print(f"Converting vocabulary to optimized format at {optimized_vocabulary}...")
    print(f"Source vocabulary: {source_vocabulary}")

    with open(optimized_vocabulary, "wb") as f:
        for word, index in source_vocabulary.items():
            # Write the word as a null-terminated byte string
            f.write(word.encode('utf-8') + b'\0')
            # Write the index as a 4-byte little-endian integer
            f.write(pack('<I', index))

    return optimized_vocabulary


def _load_vocabulary(config_file: Path) -> dict:
    """
    Loads the vocabulary from the PaddleOCR model configuration file.

    Args:
        config_file (Path): Path to the configuration file of the Piper model.

    Returns:
        dict: A dictionary containing the vocabulary with character->index mapping.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        post_process_map = config.get("PostProcess", {})
        character_dictionary = post_process_map.get("character_dict", [])
        return {character: idx for idx, character in enumerate(character_dictionary)}
