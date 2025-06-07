from pathlib import Path
from yaml import dump

from .typing import Decoder, decoder_serializer

def create_decoder_file(model_path: Path):
    """
    Creates a decoder configuration file for the specified model path if it does not already exist.
    This function checks if the decoder file exists, and if not, it creates a new one with default settings.

    Args:
        model_path (Path): Path to the directory where the decoder file will be created.
    """
    decoder_file = model_path / "decoder.yml"

    if decoder_file.exists():
        return

    content = Decoder(
        relative_paths=True,
        models=_find_model_files(model_path),
        vocabs=_find_vocab_files(model_path),
        beam_size=6,
        normalize=1,
        word_penalty=0,
        mini_batch=1,
        maxi_batch=1,
        maxi_batch_sort="src"
    )

    _save_decoder_file(model_path, content)

def _save_decoder_file(model_path: Path, content: Decoder):
    """
    Saves the decoder configuration to a YAML file.

    Args:
        model_path (Path): Path to the directory where the decoder file will be saved.
        content (Decoder): The content to be saved in the decoder file.
    """
    decoder_file = model_path / "decoder.yml"

    with open(decoder_file, 'w', encoding='utf-8') as file:
        dump(decoder_serializer(content), file, allow_unicode=True, sort_keys=False)

def _find_model_files(model_path: Path) -> list[str]:
    """
    Finds all model file names in the specified directory.

    Args:
        model_path (Path): Path to the directory containing the pre-trained model.

    Returns:
        list[str]: List of paths to the model files found in the directory.
    """
    files = list(model_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No model files found in the directory: {model_path}")

    return [file.name for file in files if file.is_file()]

def _find_vocab_files(model_path: Path) -> list[str]:
    """
    Finds all vocabulary files names in the specified directory.

    Args:
        model_path (Path): Path to the directory containing the pre-trained model.

    Returns:
        list[str]: List of paths to the vocabulary files found in the directory.
    """
    files = list(model_path.glob("*.yml")) + list(model_path.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No vocabulary files found in the directory: {model_path}")

    return [file.name for file in files if file.is_file()]
