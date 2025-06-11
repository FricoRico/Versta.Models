import torch
import json

from huggingface_hub import hf_hub_download, snapshot_download
from kokoro import KModel
from pathlib import Path
from shutil import copyfile
from struct import pack
from io import BytesIO
from numpy import save
from onnx import load
from onnx.checker import check_model
from torch import randint, LongTensor, FloatTensor, randn, onnx

VOICES = [
    "af_heart",
    "am_puck",
    "ef_dora",
    "em_alex",
    "ff_siwis",
    "hf_beta",
    "hm_omega",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "zf_xiaobei",
    "zm_yunjian"
]

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

    model_file = export_path / "model.onnx"
    config_file = export_path / "config.json"

    model_config = hf_hub_download(model_name, "config.json")
    model_checkpoints = hf_hub_download(model_name, KModel.MODEL_NAMES[model_name])

    kmodel = KModel(config=model_config, model=model_checkpoints, disable_complex=True)

    copyfile(model_config, config_file)

    model_file = _convert_model(kmodel, model_file)

    _extract_vocab(export_path, config_file)
    _convert_voices(export_path / "voices", Path(model_path))

    _set_model_type(config_file, "albert")

    return model_file


def _convert_model(kmodel: KModel, export_path: Path) -> Path:
    """
    Converts the Kokoro model to ONNX format and saves it in the export directory.

    Args:
        kmodel (KModel): The Kokoro model to convert.
        export_path (Path): Path to the file where the ONNX model will be saved.
    Returns:
    """
    print("Skipping Kokoro model conversion, as it is currently broken for unknown reasons.")
    return export_path

    model = KModelForONNX(kmodel).eval()

    input_ids = randint(1, 100, (48,)).numpy()
    input_ids = LongTensor([[0, *input_ids, 0]])
    style = randn(1, 256)
    speed = randint(1, 10, (1,)).numpy()
    speed = FloatTensor([*speed])

    onnx.export(
        model,
        args=(input_ids, style, speed),
        f=export_path,
        export_params=True,
        verbose=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {1: "sequence_length"},
            "waveform": {1: "num_samples"},
        },
        do_constant_folding=True,
    )

    check_model(
        model=load(export_path),
        full_check=True,
        check_custom_domain=True
    )

    return export_path


def _set_model_type(config_file: Path, model_type: str):
    """
    Sets the model type in the configuration file.

    Args:
        config_file (Path): Path to the configuration file of the Kokoro model.
        model_type (str): The type of the model to set.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["model_type"] = model_type

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


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
        if file.stem not in VOICES:
            continue

        voice_file = export_dir / f"{file.stem}.npy"

        if not voice_file.exists():
            export_dir.mkdir(parents=True, exist_ok=True)

        with open(file, "rb") as f:
            content = BytesIO(f.read())

        data = torch.load(content, weights_only=True).numpy()

        with open(voice_file, "wb") as f:
            save(f, data)


class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
            self,
            input_ids: torch.LongTensor,
            ref_s: torch.FloatTensor,
            speed: float = 1
    ) -> torch.Tensor:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform.unsqueeze(0)
