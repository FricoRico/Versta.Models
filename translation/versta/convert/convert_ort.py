from onnxruntime.tools.convert_onnx_models_to_ort import (
    OptimizationStyle,
    convert_onnx_models_to_ort
)
from pathlib import Path
from typing import TypedDict

class ORTFiles(TypedDict):
    encoder: Path
    decoder: Path

def convert_model_to_ort(input_dir: Path, output_dir: Path) -> ORTFiles:
    """
    Converts the model from ONNX format to ORT format.

    Args:
        input_dir (Path): Path to the directory where the ONNX model are imported from.
        output_dir (Path): Path to the directory where the ORT model will be saved.

    Returns:
        ORTFiles: Dictionary containing the file paths for the encoder and decoder ORT files.
    """
    convert_onnx_models_to_ort(
        input_dir,
        output_dir,
        optimization_styles=[OptimizationStyle.Runtime],
        target_platform="arm",
        enable_type_reduction=True,
    )

    # Get the file paths for the encoder and decoder ORT files
    return _get_files_from_output(output_dir)

def _get_files_from_output(output_dir: Path) -> ORTFiles:
    """
    Looks in the specified output directory for files to find the output encoder and decoder files.

    Args:
        output_dir (str): The directory to search for files.

    Returns:
        ORTFiles: Dictionary containing the file paths for the encoder and decoder ORT files.
    """
    # Check that the directory exists
    if not output_dir.is_dir():
        raise FileNotFoundError(f"The directory {output_dir} does not exist.")

    # List to hold matching files
    ort_files = ORTFiles(encoder=None, decoder=None)

    # Iterate over all files in the directory
    for path in output_dir.glob('*'):
        file = Path(path).name

        if "encoder_model" in file:
            ort_files["encoder"] = file
        elif "decoder_model_merged" in file:
            ort_files["decoder"] = file

    # Check if any required file is still None, raise an error if it is
    missing_files = [key for key, value in ort_files.items() if value is None]

    if missing_files:
        raise FileNotFoundError(f"Missing required ORT output files: {', '.join(missing_files)}")

    return ort_files