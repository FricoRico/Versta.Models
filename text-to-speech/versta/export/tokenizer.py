from pathlib import Path
from shutil import copyfile

from .typing import TokenizerFiles

def save_tokenizer(input_dir: Path, output_dir: Path) -> TokenizerFiles:
    """
    Saves the tokenizer files from the source directory to the export directory.

    Args:
        input_dir (Path): Path to the directory containing the tokenizer files.
        output_dir (Path): Path to the directory where tokenizer files will be saved.

    Returns:
        TokenizerFiles: Dictionary containing the file paths for the tokenizer configuration, vocabulary, source, and target files.
    """
    vocabulary_file = input_dir / "vocab.bin"

    if not vocabulary_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocabulary_file}.")

    copyfile(vocabulary_file, output_dir / "vocab.bin")

    tokenizer_files = TokenizerFiles(vocabulary="vocab.bin")

    return tokenizer_files