import json
from pathlib import Path
from struct import pack

from .typing import TokenizerFiles


def save_tokenizer(input_dir: Path, output_dir: Path) -> TokenizerFiles | None:
    """
     Extracts the vocabulary from the Piper model config.json and saves it in the export directory.

     Args:
        input_dir (Path): Path to the directory containing the tokenizer files.
        output_dir (Path): Path to the directory where tokenizer files will be saved.

     Returns:
         TokenizerFiles: Dictionary containing the file paths for the tokenizer vocabulary files.
     """
    vocabulary_file = _save_vocabulary(input_dir, output_dir)

    if vocabulary_file is None:
        return None

    tokenizer_files = TokenizerFiles(vocabulary="vocab.bin")

    return tokenizer_files

def _save_vocabulary(input_dir: Path, output_dir: Path) -> Path | None:
    """
    Extracts the vocabulary from the PaddleOCR config file.

     Args:
        input_dir (Path): Path to the directory containing the tokenizer files.
        output_dir (Path): Path to the directory where tokenizer files will be saved.

     Returns:
         TokenizerFiles: Path to the created vocab.bin file.
    """
    config_file = input_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    source_vocabulary = _load_vocabulary(config_file)

    if not source_vocabulary:
        return None

    optimized_vocabulary = output_dir / "vocab.bin"

    print(f"Converting vocabulary to optimized format at {optimized_vocabulary}...")

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
