from optimum.exporters.onnx import main_export
from onnxruntime_genai.models.builder import create_model
from pathlib import Path
from os import rename
import subprocess

def convert_model_to_onnx(model_name: str, export_dir: Path):
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_name (str): Name of the pre-trained model.
        export_dir (Path): Path to the directory where the ONNX model will be saved.
    """
    task = "text2text-generation-with-past"

    print(f"Exporting {model_name} to ONNX format...")

    main_export(
        model_name,
        output=export_dir,
        task=task,
        framework="pt",
        opset=20,
        library_name="transformers",
    )

    encoder_command = [
        "olive",
        "auto-opt",
        "--model_name_or_path", export_dir / "encoder_model.onnx",
        "--output_path", export_dir,
        "--device", "cpu",
        "--provider", "CPUExecutionProvider",
        "--precision", "fp32",
    ]

    decoder_command = [
        "olive",
        "auto-opt",
        "--model_name_or_path", export_dir / "decoder_model_merged.onnx",
        "--output_path", export_dir,
        "--device", "cpu",
        "--provider", "CPUExecutionProvider",
        "--precision", "fp32",
    ]

    subprocess.run(encoder_command, capture_output=False, text=True)
    rename(export_dir / "model.onnx", export_dir / "encoder_model_olive.onnx")

    subprocess.run(decoder_command, capture_output=False, text=True)
    rename(export_dir / "model.onnx", export_dir / "decoder_model_merged_olive.onnx")
