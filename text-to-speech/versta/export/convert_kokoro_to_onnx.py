import torch
import json

from huggingface_hub import hf_hub_download, snapshot_download
from kokoro import KModel
from kokoro.model import KModelForONNX
from pathlib import Path
from shutil import copyfile
from struct import pack
from io import BytesIO

from numpy import save
from onnx import load
from onnx.checker import check_model
from torch import randint, LongTensor, randn, onnx

def convert_kokoro_to_onnx(model_name: str, export_path: Path) -> Path:
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_name (str): Name of the pre-trained model.
        export_path (Path): Path to the directory where the ONNX model will be saved.
    Returns:
        Path: The path to the exported ONNX model.
    """
    output = snapshot_download(model_name)
    model_path = Path(output)

    model_config = hf_hub_download(model_name, "config.json")
    model_checkpoints = hf_hub_download(model_name, KModel.MODEL_NAMES[model_name])

    kokoro_model = KModel(config=model_config, model=model_checkpoints, disable_complex=True)
    model = KModelForONNX(kokoro_model).eval()

    model_file = export_path / "model.onnx"
    config_file = export_path / "config.json"

    input_ids = randint(1, 100, (48,)).numpy()
    input_ids = LongTensor([[0, *input_ids, 0]])
    style = randn(1, 256)
    speed = randint(1, 10, (1,)).int()

    copyfile(model_config, config_file)
    onnx.export(
        model,
        args=(input_ids, style, speed),
        f=model_file,
        export_params=True,
        verbose=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "duration"],
        opset_version=20,
        dynamic_axes={
            "input_ids": {1: "sequence_length"},
            "waveform": {0: "num_samples"},
        },
        do_constant_folding=True,
    )

    onnx_model = load(model_file)
    check_model(onnx_model)

    _extract_vocab(export_path, config_file)
    _convert_voices(export_path / "voices", Path(model_path))

    return model_file

def _extract_vocab(export_dir: Path, config_file: Path) -> Path:
    """
    Extracts the vocabulary from the Kokoro model and saves it in the export directory.
    Args:
        export_dir (Path): Path to the directory where the vocabulary will be saved.
        config_file (Path): Path to the configuration file of the Kokoro model.
    """

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    source_vocabulary = _load_vocabulary(config_file)
    optimized_vocabulary = export_dir / "vocab.bin"

    with open(optimized_vocabulary, "wb") as f:
        for word, index in source_vocabulary.items():
            # Write the word as a null-terminated byte string
            f.write(word.encode('utf-8') + b'\0')
            # Write the index as a 4-byte little-endian integer
            f.write(pack('<I', index))

    return optimized_vocabulary

def _load_vocabulary(config_file: Path) -> dict:
    """
    Loads the vocabulary from the Kokoro model configuration file.

    Args:
        config_file (Path): Path to the configuration file of the Kokoro model.

    Returns:
        dict: A dictionary containing the vocabulary.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        return config.get("vocab", {})


def _convert_voices(export_dir: Path, model_path: Path):
    """
    Converts the voices from the Kokoro model to the export directory.

    Args:
        export_dir (Path): Path to the directory where the voices will be saved.
    """

    voices_path = model_path / "voices"
    if not voices_path.exists():
        raise FileNotFoundError(f"Voices directory not found at {voices_path}.")

    for file in voices_path.glob("*.pt"):
        voice_file = export_dir / f"{file.stem}.npy"
        if not voice_file.exists():
            export_dir.mkdir(parents=True, exist_ok=True)

        with open(file, "rb") as f:
            content = BytesIO(f.read())

        data = torch.load(content, weights_only=True).numpy()

        with open(voice_file, "wb") as f:
            save(f, data)
