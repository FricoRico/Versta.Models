from transformers import MarianMTModel, MarianTokenizer
import torch

from transformers.models.marian.convert_marian_to_pytorch import convert
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import convert as convert_tatoeba
from pathlib import Path
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

def convert_model_to_pt(model_path: Path, export_dir: Path, type: str) -> Path:
    """
    Converts the specified pre-trained model to PyTorch format and saves it in the export directory.

    Args:
        model_path (Path): Path to downloaded pre-trained numpy model.
        export_dir (Path): Path to the directory where the PyTorch model will be saved.
        type (str): Type of the pre-trained model to convert (e.g., "opus" "opus-tatoeba").
    """
    model_path = Path("." / model_path)
    export_dir = "." / export_dir

    print(f"Converting {model_path} to PyTorch format...")

    if type == "opus":
        _convert_model_to_pt(model_path, export_dir)
    elif type == "opus-tatoeba":
        _convert_tatoeba_model_to_pt(model_path, export_dir)
    else:
        raise ValueError(f"Invalid model type: {type}")

    _run_translation_test(export_dir)
    
    return export_dir
    
def _convert_model_to_pt(model_path: Path, export_dir: Path):
    """
    Converts the specified pre-trained model to PyTorch format and saves it in the export directory.

    Args:
        model_path (Path): Path to the pre-trained model.
        export_dir (Path): Path to the directory where the PyTorch model will be saved.
    """
    convert(Path(model_path), export_dir)

def _convert_tatoeba_model_to_pt(model_path: Path, export_dir: Path):
    """
    Converts the specified Tatoeba pre-trained model to PyTorch format and saves it in the export directory.

    Args:
        model_path (Path): Path to downloaded pre-trained numpy model.
        export_dir (Path): Path to the directory where the PyTorch model will be saved.
    """
    convert_tatoeba(Path(model_path), export_dir)

def _run_translation_test(export_dir: Path):
    """
    Runs a simple test translation to validate the converted PyTorch model

    Args:
        export_dir (Path): Path to the directory where the PyTorch model is saved.
    """

    model = MarianMTModel.from_pretrained(export_dir)
    tokenizer = MarianTokenizer.from_pretrained(export_dir)

    src_text = "Israeli Prime Minister Netanyahu is consulting with his defense top over Hamas’s decision to stop the release of hostages indefinitely. Tomorrow morning the Israeli security cabinet will meet, a meeting that would originally be held in the afternoon. It is unclear what comes out. Defense Minister Katz has brought the Israeli army in Gaza to the highest state of readiness."
    translated = model.generate(**tokenizer(src_text, return_tensors='pt'))

    print(f"Translated: {tokenizer.decode(translated[0], skip_special_tokens=True)}")


