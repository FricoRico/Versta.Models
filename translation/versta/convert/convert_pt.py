from pathlib import Path

from transformers import MarianMTModel, MarianTokenizer

from transformers.models.marian.convert_marian_to_pytorch import convert
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import convert as convert_tatoeba

def convert_model_to_pt(model_path: Path, export_dir: Path, type: str) -> Path:
    """
    Converts the specified pre-trained model to PyTorch format and saves it in the export directory.

    Args:
        model_path (Path): Path to downloaded pre-trained numpy model.
        export_dir (Path): Path to the directory where the PyTorch model will be saved.
        type (str): Type of the pre-trained model to convert (e.g., "opus" "opus-tatoeba").
    Returns:
        Path: Path to the directory where the PyTorch model is saved.
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

    src_text = "우크라이나 젤렌스키 대통령은 러시아가 \"드디어 전쟁 종식을 고려하고 있다\"는 것은 긍정적인 신호라고 하면서도, 협상이 아닌 휴전이 다음 단계가 되어야 한다고 강조했습니다. 그는 X에 올린 메시지에서 어젯밤 푸틴 대통령의 제안에 답했습니다. 언론에 출연한 젤렌스키 대통령은 다음 주 목요일 터키 이스탄불에서 협상을 제안했습니다."
    src_text = "Ukrainian President Zelensky calls it a positive sign that the Russians are \"finally thinking about ending the war\", but states that not negotiations but a ceasefire should be the next step. In a message on X he responds to a proposal from Putin last night. In a media appearance, the Russian president proposed negotiations in Istanbul, Turkey, next Thursday."
    translated = model.generate(**tokenizer(src_text, return_tensors='pt'))

    print(f"Translated: {tokenizer.decode(translated[0], skip_special_tokens=True)}")


