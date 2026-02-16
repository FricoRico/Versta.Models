from pathlib import Path

from paddlex.inference.utils.model_paths import get_model_paths
from paddle2onnx import export

def convert_model_to_onnx(model_path: Path, converted_dir: Path):
    """
    Exports the specified pre-trained model to ONNX format and saves it in the converted directory.
    The model is exported in FP16 format for optimal performance.

    Args:
        model_path (Path): Path to the pre-trained model.
        converted_dir (Path): Path to the directory where the FP16 ONNX model will be saved.
    Returns:
        Path: The path to the exported FP16 ONNX model.
    """
    print(f"Exporting {model_path} to ONNX format (FP16)...")

    input_files = get_model_paths(model_path)
    output_file = converted_dir / "model.onnx"

    export(
        model_filename=input_files["paddle"][0].as_posix(),
        params_filename=input_files["paddle"][1].as_posix(),
        save_file=output_file,
        opset_version=14,
        auto_upgrade_opset=True,
        export_fp16_model=True,
    )

    print(f"ONNX model exported to {output_file}")
    return output_file
