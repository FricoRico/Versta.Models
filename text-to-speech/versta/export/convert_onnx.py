from huggingface_hub import hf_hub_download
from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX
from pathlib import Path
from shutil import copyfile
import torch


def convert_model_to_onnx(model_name: str, export_dir: Path):
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_name (str): Name of the pre-trained model.
        export_dir (Path): Path to the directory where the ONNX model will be saved.
    """
    print(f"Exporting {model_name} to ONNX format...")

    modelConfig = hf_hub_download(model_name, "config.json")
    modelCheckpoints = hf_hub_download(model_name, "kokoro-v1_1-zh.pth")

    kokoroModel = KModel(config=modelConfig, model=modelCheckpoints, disable_complex=True)
    model = KModelForONNX(kokoroModel).eval()

    file = export_dir / "model.onnx"

    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.randint(1, 10, (1,)).int()

    copyfile(modelConfig, export_dir / "config.json")
    torch.onnx.export(
        model,
        args=(input_ids, style, speed),
        f=file,
        export_params=True,
        verbose=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "duration"],
        opset_version=20,
        dynamic_axes={
            "input_ids": {1: "input_ids_len"},
            "waveform": {0: "num_samples"},
        },
        do_constant_folding=True,
    )
