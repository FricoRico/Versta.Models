from pathlib import Path
from yaml import dump

from sentencepiece import SentencePieceProcessor

def convert_vocabulary(model_path: Path):
    """
    Converts the vocabulary of the specified pre-trained model to a format compatible with PyTorch.

    Args:
        model_path (Path): Path to the pre-trained model.
    """
    if _vocab_exists(model_path):
        return

    print(f"Converting vocabulary for {model_path}...")
    spm_file = _find_sentence_piece_file(model_path)

    processor = SentencePieceProcessor()
    processor.load(spm_file.as_posix())

    vocab = [processor.id_to_piece(i) for i in range(processor.get_piece_size())]
    _save_vocab_as_yaml(vocab, model_path / "vocab.yml")

def _save_vocab_as_yaml(vocab: list, file_path: Path):
    """
    Saves the vocabulary to a YAML file.

    Args:
        vocab (list): List of vocabulary tokens.
        file_path (Path): Path where the vocabulary file will be saved.
    """
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}
    with open(file_path, 'w', encoding='utf-8') as file:
        dump(vocab_dict, file, allow_unicode=True, sort_keys=False)


def _vocab_exists(model_dir) -> bool:
    """
    Finds the vocabulary file in the specified model directory.

    Args:
        model_dir (Path): Path to the directory containing the pre-trained model.
    Returns:
        bool: True if a vocabulary file exists, False otherwise.
    """
    return any(model_dir.glob("*vocab.yml")) or any(model_dir.glob("*vocab.json"))

def _find_sentence_piece_file(model_dir: Path) -> Path | None:
    """
    Finds the sentence piece file in the specified model directory.

    Args:
        model_dir (Path): Path to the directory containing the pre-trained model.

    Returns:
        Path: Path to the sentence piece file if found, otherwise None.
    """
    for file in model_dir.glob("*.spm"):
        return file

    return None