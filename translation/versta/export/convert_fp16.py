from pathlib import Path

import onnx
import onnxslim
from onnxconverter_common import float16


def convert_to_fp16(input_dir: Path, model_filename: str, output_dir: Path) -> str:
    """
    Converts an ONNX model to FP16 format and optimizes it with onnxslim.

    Args:
        input_dir (Path): Directory containing the input ONNX model.
        model_filename (str): Name of the ONNX model file to convert.
        output_dir (Path): Directory where the FP16 model will be saved.

    Returns:
        str: The filename of the converted FP16 model.
    """
    input_path = input_dir / model_filename

    base_name = model_filename.replace(".onnx", "_fp16.onnx")
    output_path = output_dir / base_name

    print(f"Loading ONNX model from {input_path}...")
    model = onnx.load(str(input_path))

    print(f"Converting {model_filename} to FP16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=False,
    )

    print(f"Optimizing {model_filename} with onnxslim...")
    model_simplified = onnxslim.slim(model_fp16)

    print(f"Saving FP16 model to {output_path}...")
    onnx.save(model_simplified, str(output_path))

    print(f"FP16 conversion complete for {model_filename}!")
    return base_name


def convert_models_to_fp16(
    input_dir: Path,
    output_dir: Path,
    encoder_filename: str = "encoder_model.onnx",
    decoder_filename: str = "decoder_with_past_model.onnx",
) -> tuple[str, str]:
    """
    Converts both encoder and decoder models to FP16 format.

    Args:
        input_dir (Path): Directory containing the input ONNX models.
        output_dir (Path): Directory where the FP16 models will be saved.
        encoder_filename (str): Name of the encoder ONNX file.
        decoder_filename (str): Name of the decoder with past ONNX file.

    Returns:
        tuple[str, str]: Filenames of the converted encoder and decoder FP16 models.
    """
    encoder_fp16 = convert_to_fp16(input_dir, encoder_filename, output_dir)
    decoder_fp16 = convert_to_fp16(input_dir, decoder_filename, output_dir)

    return encoder_fp16, decoder_fp16
