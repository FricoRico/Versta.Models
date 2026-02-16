from pathlib import Path
import onnxsim
from onnx import load_model, save_model

input_shapes = {
    "detector": [1, 3, 640, 640],    # [batch, channels, height, width]
    "recognizer": [1, 3, 48, 640],   # [batch, channels, height, width]
}

def simplify_model(input_path: Path, output_path: Path, module: str = "recognizer"):
    """
    Simplifies an ONNX model using onnxsim.

    Args:
        input_path (Path): Path to the input ONNX model.
        output_path (Path): Path where the simplified ONNX model will be saved.
        module (str): Type of OCR module - 'detector' or 'recognizer'.
                     Used to determine input shape for onnxsim.
    """
    if module not in input_shapes:
        raise ValueError(f"Unknown module type: {module}. Must be one of: {list(input_shapes.keys())}")

    input_shape = input_shapes[module]

    print(f"Loading ONNX model from {input_path}...")
    model = load_model(input_path.as_posix())

    print(f"Simplifying model with onnxsim (input shape: {input_shape})...")
    model_simplified, check = onnxsim.simplify(
        model,
        test_input_shapes={model.graph.input[0].name: input_shape}
    )
    if not check:
        raise RuntimeError("onnxsim failed to simplify the model")

    print(f"Saving simplified model to {output_path}...")
    save_model(model_simplified, output_path.as_posix())

    print("Simplification complete!")
