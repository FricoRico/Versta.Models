from pathlib import Path
from typing import Optional, Literal
import tempfile
import shutil

from paddle2onnx import export


def convert_paddle_to_tflite(
    model_path: Path,
    output_dir: Path,
    quantization: Literal["none", "dynamic_int8", "fp16"] = "dynamic_int8",
    intermediates_dir: Optional[Path] = None,
    keep_intermediates: bool = False,
) -> Path:
    """
    Convert PaddleOCR model to TFLite format.

    Pipeline: Paddle → ONNX (intermediates) → TFLite → cleanup (optional)

    Args:
        model_path: Path to Paddle model directory
        output_dir: Directory for output .tflite file
        quantization: Quantization type ("dynamic_int8", "fp16", or "none")
        intermediates_dir: Directory for intermediate files (if None, uses temp dir)
        keep_intermediates: Whether to keep intermediate files

    Returns:
        Path to generated .tflite file
    """
    print(f"Converting Paddle model to TFLite with {quantization} quantization...")

    # 1. Setup intermediates directory
    if intermediates_dir is not None:
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        converted_dir = intermediates_dir / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = converted_dir / "model.onnx"
        temp_dir_obj = None
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir_obj.name)
        intermediates_dir = temp_path / "intermediates"
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        converted_dir = intermediates_dir / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = converted_dir / "model.onnx"

    try:
        # 2. Convert Paddle to ONNX
        print("  Step 1/3: Converting Paddle to ONNX...")
        _convert_paddle_to_onnx(model_path, onnx_path)

        # 3. Convert ONNX to TFLite using native API
        print("  Step 2/3: Converting ONNX to TFLite...")
        _convert_onnx_to_tflite(onnx_path, converted_dir)

        # 4. Apply quantization if requested
        output_dir.mkdir(parents=True, exist_ok=True)
        final_tflite_path = output_dir / "model.tflite"
        
        if quantization != "none":
            print(f"  Step 3/3: Applying {quantization} quantization...")
            _apply_quantization(converted_dir, final_tflite_path, quantization)
        else:
            print("  Step 3/3: Skipping quantization, copying to output...")
            # Find and copy the generated TFLite model
            source_tflite = converted_dir / "model_float32.tflite"
            if not source_tflite.exists():
                raise FileNotFoundError(
                    f"TFLite model not found at {source_tflite}. "
                    f"Files in directory: {list(converted_dir.glob('*.tflite'))}"
                )
            shutil.copy(str(source_tflite), str(final_tflite_path))
        
        # 5. Cleanup intermediates if requested
        if not keep_intermediates and temp_dir_obj is None:
            print("  Cleaning up intermediate files...")
            shutil.rmtree(intermediates_dir, ignore_errors=True)
            
    finally:
        # Cleanup temp directory if used
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    print(f"TFLite model saved to: {final_tflite_path}")
    return final_tflite_path


def _convert_paddle_to_onnx(model_path: Path, output_path: Path):
    """Convert Paddle model to ONNX format."""
    from paddlex.inference.utils.model_paths import get_model_paths

    input_files = get_model_paths(model_path)

    export(
        model_filename=input_files["paddle"][0].as_posix(),
        params_filename=input_files["paddle"][1].as_posix(),
        save_file=output_path,
        opset_version=14,
        auto_upgrade_opset=True,
        enable_onnx_checker=True,
        enable_optimize=True,
        export_fp16_model=False,
    )


def _convert_onnx_to_tflite(onnx_path: Path, output_dir: Path):
    """Convert ONNX to TFLite format using native onnx2tf API.
    
    Args:
        onnx_path: Path to input ONNX model
        output_dir: Directory where TFLite model(s) will be saved
    """
    from onnx2tf import convert

    convert(
        input_onnx_file_path=onnx_path.as_posix(),
        output_folder_path=output_dir.as_posix(),
        copy_onnx_input_output_names_to_tflite=True,
        overwrite_input_shape=["x:1,3,48,960"],
        # disable_group_convolution=True,
        # disable_strict_mode=True,
        # optimization_for_gpu_delegate=True,
        # verbosity='info',
    )


def _apply_quantization(source_path: Path, dest_path: Path, quantization: str):
    """Apply post-training quantization to TFLite model.
    
    Args:
        source_path: Path to saved_model directory
        dest_path: Path where quantized model will be saved
        quantization: Type of quantization ("dynamic_int8" or "fp16")
    """
    import tensorflow as tf

    # Create converter from saved model - convert Path to string
    converter = tf.lite.TFLiteConverter.from_saved_model(str(source_path))

    if quantization == "dynamic_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # Convert with quantization
    quantized_model = converter.convert()

    # Write quantized model to destination
    dest_path.write_bytes(quantized_model)
