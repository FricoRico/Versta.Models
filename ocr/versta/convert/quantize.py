from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(export_dir: Path, model_filename: str, quantization_dir: Path):
    """
    Quantizes a specific ONNX model file and saves the quantized model to the given directory.

    Args:
        export_dir (Path): Path to the directory where the ONNX model is stored.
        model_filename (str): Name of the ONNX model file to quantize (e.g., "encoder_model.onnx").
        quantization_dir (Path): Path to the directory where the quantized model will be saved.
        config (AutoQuantizationConfig): Configuration for the quantization process.
    """
    model_path = export_dir / model_filename
    quantized_model_path = quantization_dir / f"{model_path.stem}.quantized.onnx"

    print(f"Quantizing the model {model_filename} to uint8...")

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(quantized_model_path),
        weight_type=QuantType.QUInt8
    )

