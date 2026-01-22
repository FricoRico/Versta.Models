"""
Graph optimization for ONNX models using ONNX-Slim.
"""

from pathlib import Path
import onnxslim


def optimize_model(
    model_path: Path,
    output_path: Path,
    num_passes: int = 2,
    enable_fusion: bool = True,
) -> Path:
    """
    Optimize ONNX model graph using ONNX-Slim.

    Runs multiple optimization passes to reduce model size and improve inference efficiency.
    ONNX-Slim removes redundant operations, folds constants, and simplifies the graph structure.

    Args:
        model_path: Path to input ONNX model
        output_path: Path for optimized ONNX model
        num_passes: Number of optimization passes to run (default: 2)
        enable_fusion: Enable operator fusion patterns (default: True)

    Returns:
        Path to optimized model
    """
    print(f"\n{'=' * 70}")
    print(f"Optimizing ONNX model with ONNX-Slim")
    print(f"{'=' * 70}")
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")
    print(f"Number of passes: {num_passes}")
    print(f"Operator fusion: {'enabled' if enable_fusion else 'disabled'}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get original model size
    original_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\nOriginal model size: {original_size:.2f} MB")

    # Run optimization passes
    current_input = model_path
    current_output = output_path

    for pass_num in range(1, num_passes + 1):
        print(f"\n--- Pass {pass_num}/{num_passes} ---")

        # For intermediate passes, use a temporary file
        if pass_num < num_passes:
            current_output = output_path.parent / f"temp_pass{pass_num}.onnx"
        else:
            current_output = output_path

        # Run onnxslim
        print(f"Running onnxslim...")
        if enable_fusion:
            onnxslim.slim(
                str(current_input),
                str(current_output),
                skip_fusion_patterns=False,
            )
        else:
            onnxslim.slim(
                str(current_input),
                str(current_output),
                skip_fusion_patterns=True,
            )

        # Report size after this pass
        pass_size = current_output.stat().st_size / (1024 * 1024)  # MB
        reduction = (
            (current_input.stat().st_size - current_output.stat().st_size)
            / current_input.stat().st_size
        ) * 100
        print(f"After pass {pass_num}: {pass_size:.2f} MB ({reduction:+.1f}%)")

        # Clean up intermediate input if it was temporary
        if pass_num > 1 and current_input != model_path:
            current_input.unlink()

        # Set output of this pass as input for next pass
        current_input = current_output

    # Final size comparison
    optimized_size = output_path.stat().st_size / (1024 * 1024)  # MB
    total_reduction = ((original_size - optimized_size) / original_size) * 100

    print(f"\n{'=' * 70}")
    print(f"Optimization complete!")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Optimized size: {optimized_size:.2f} MB")
    print(f"  Total reduction: {total_reduction:.1f}%")
    print(f"  Saved: {original_size - optimized_size:.2f} MB")
    print(f"{'=' * 70}\n")

    return output_path
