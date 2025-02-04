
from transformers.models.marian.convert_marian_to_pytorch import convert
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import convert as convert_tatoeba
from pathlib import Path

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

    _normalize_sentence_piece_models(model_path)

    if type == "opus":
        _convert_model_to_pt(model_path, export_dir)
    elif type == "opus-tatoeba":
        _convert_tatoeba_model_to_pt(model_path, export_dir)
    else:
        raise ValueError(f"Invalid model type: {type}")
    
    return export_dir
    
def _convert_model_to_pt(model_path: str, export_dir: str):
    """
    Converts the specified pre-trained model to PyTorch format and saves it in the export directory.

    Args:
        model_path (Path): Path to the pre-trained model.
        export_dir (Path): Path to the directory where the PyTorch model will be saved.
    """
    convert(Path(model_path), export_dir)

def _convert_tatoeba_model_to_pt(model_path: str, export_dir: str):
    """
    Converts the specified Tatoeba pre-trained model to PyTorch format and saves it in the export directory.

    Args:
        model_path (Path): Path to downloaded pre-trained numpy model.
        export_dir (Path): Path to the directory where the PyTorch model will be saved.
    """
    convert_tatoeba(Path(model_path), export_dir)

def _normalize_sentence_piece_models(model_path: Path):
    """
    Normalize the SentencePiece models for the specified pre-trained model. Sometimes the SentencePiece models
    are named differently (eg. bpe file format), so this function will rename them to the expected .spm format.

    Args:
        model_path (Path): Path to the pre-trained model.
    """
    for file in model_path.glob("*.bpe"):
        print(f"Renaming {file} to {file.with_suffix('.spm')}")
        file.rename(file.with_suffix(".spm"))