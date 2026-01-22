from pathlib import Path

from paddlex.inference.utils.model_paths import get_model_paths
from paddle2onnx import export, paddle2onnx_cpp2py_export


def convert_model_to_onnx(
    model_path: Path,
    export_dir: Path,
    prune_ratio: float = 0.0,
    precision: str = "fp16",
) -> Path:
    """
    Exports the specified pre-trained model to ONNX format.

    Converts Paddle model to ONNX FP32 or FP16 format based on precision parameter.
    No optimization is performed here.

    Args:
        model_path: Path to the pre-trained Paddle model.
        export_dir: Path to the directory where the ONNX model will be saved.
        prune_ratio: Fraction of channels to prune (0.0 to 0.5, default 0.0 = no pruning)
        precision: "fp32" or "fp16" (default: "fp16")
    """
    export_precision = "FP16" if precision == "fp16" else "FP32"
    print(f"\n{'=' * 70}")
    print(f"Converting model to ONNX format ({export_precision})")
    print(f"{'=' * 70}")
    print(f"Model path: {model_path}")
    print(f"Export directory: {export_dir}")
    print(f"Pruning ratio: {prune_ratio:.1%}")
    print(f"  Note: Optimization disabled before quantization")

    input_files = get_model_paths(model_path)
    output_fp32_file = export_dir / "model.onnx"

    print(f"\n{'=' * 70}")
    print(f"Exporting to FP32 ONNX format...")
    print(f"{'=' * 70}")

    export(
        model_filename=input_files["paddle"][0].as_posix(),
        params_filename=input_files["paddle"][1].as_posix(),
        save_file=output_fp32_file,
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
        export_fp16_model=False,
        optimize_tool="onnxoptimizer",
    )

    fp32_size = output_fp32_file.stat().st_size / (1024 * 1024)
    print(f"\nFP32 model saved: {fp32_size:.2f} MB")

    if precision == "fp16":
        print(f"\n{'=' * 70}")
        print(f"Converting to FP16...")
        print(f"{'=' * 70}")

        output_fp16_file = export_dir / "model_fp16_temp.onnx"
        paddle2onnx_cpp2py_export.convert_to_fp16(
            output_fp32_file.as_posix(), output_fp16_file.as_posix(), True
        )

        output_fp32_file.unlink()
        output_fp16_file.rename(output_fp32_file)
        print(f"\nRemoved intermediate FP32 model")

        fp16_size = output_fp32_file.stat().st_size / (1024 * 1024)
        size_reduction = ((fp32_size - fp16_size) / fp32_size) * 100

        print(f"\nFP16 model saved: {fp16_size:.2f} MB")
        print(f"Size reduction from FP32: {size_reduction:.1f}%")

        print(f"\n{'=' * 70}")
        print(f"Conversion complete!")
        print(
            f"  FP16 size: {fp16_size:.2f} MB ({size_reduction:.1f}% reduction from FP32)"
        )
        print(f"  Output: {output_fp32_file}")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f"Conversion complete!")
        print(f"  FP32 size: {fp32_size:.2f} MB")
        print(f"  Output: {output_fp32_file}")
        print(f"{'=' * 70}\n")

    return output_fp32_file
