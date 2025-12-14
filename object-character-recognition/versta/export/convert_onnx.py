from pathlib import Path

from paddlex.inference.utils.model_paths import get_model_paths
from paddle2onnx import export, paddle2onnx_cpp2py_export

def convert_model_to_onnx(model_path: Path, export_dir: Path):
    """
    Exports the specified pre-trained model to ONNX format and saves it in the export directory.

    Args:
        model_path (str): Path to the pre-trained model.
        export_dir (Path): Path to the directory where the ONNX model will be saved.
    Returns:
        Path: The path to the exported ONNX model.
    """
    print(f"Exporting {model_path} to ONNX format...")

    input_files = get_model_paths(model_path)
    output_file = model_path / "model.onnx"
    output_fp16_file = export_dir / "model.onnx"

    export(
        model_filename=input_files["paddle"][0].as_posix(),
        params_filename=input_files["paddle"][1].as_posix(),
        save_file=output_file,
        opset_version=14,
        auto_upgrade_opset=True,
        dist_prim_all=False,
        verbose=False,
        enable_onnx_checker=True,
        enable_experimental_op=True,
        enable_optimize=True,
        custom_op_info=None,
        deploy_backend="onnxruntime",
        calibration_file="",
        external_file="",
        export_fp16_model=True,
        optimize_tool="onnxoptimizer",
    )

    print(f"Converting ONNX model to FP16 format...")
    paddle2onnx_cpp2py_export.convert_to_fp16(output_file.as_posix(), output_fp16_file.as_posix(), True)
