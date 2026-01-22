"""
QInt8 dynamic quantization for FP16 ONNX models using onnxruntime quantization.
Applies INT8 weights and INT16 (FP16-like) activations using dynamic quantization.
No calibration data required.
"""

from pathlib import Path


def quantize_model_to_q8_fp16_dynamic(
    model_path: Path,
    output_path: Path,
    extra_options: dict | None = None,
) -> Path:
    """
    Quantize FP16 ONNX model to QInt8 weights with QInt16 (FP16-like) activations using dynamic quantization.

    This method uses dynamic quantization where weights are quantized to INT8 and activations
    are quantized to INT16. This is equivalent to FP16 activations for WebGPU compatibility.
    No calibration data is required as quantization parameters are computed dynamically during inference.

    Args:
        model_path: Path to FP16 ONNX model
        output_path: Path for quantized QInt8 FP16 model
        extra_options: Additional options for quantization (optional)

    Returns:
        Path to quantized QInt8 FP16 model

    Raises:
        RuntimeError: If quantization fails
    """
    print(f"\n{'=' * 70}")
    print(f"QInt8 FP16 dynamic quantization")
    print(f"{'=' * 70}")
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if extra_options is None:
        extra_options = {}

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print(f"\nQuantization configuration:")
        print(f"  Weights: INT8 (QInt8)")
        print(f"  Activations: INT16 (FP16-like for WebGPU)")
        print(f"  Method: Dynamic quantization (no calibration)")
        print(f"  Operator types: MatMul, Conv, Gemm, Add, Mul (as many as supported)")

        # Calculate model size before quantization
        original_size = model_path.stat().st_size / (1024 * 1024)
        print(f"\nOriginal model size: {original_size:.2f} MB")

        # Run dynamic quantization
        print(f"\nApplying dynamic quantization...")
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            extra_options=extra_options,
        )

        # Calculate model size after quantization
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        size_reduction_percent = (
            (original_size - quantized_size) / original_size
        ) * 100

        print(f"\nQInt8 quantization complete!")
        print(f"\nModel size comparison:")
        print(f"  FP16 model: {original_size:.2f} MB")
        print(f"  QInt8 model: {quantized_size:.2f} MB")
        print(f"  Size reduction: {size_reduction_percent:.1f}%")
        print(f"  Quantized model saved to: {output_path}")
        print(f"{'=' * 70}\n")

        return output_path

    except Exception as e:
        print(f"\nQInt8 quantization failed: {e}")
        print(f"\nFull error details:")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"QInt8 quantization failed: {e}") from e
