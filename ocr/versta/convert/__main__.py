import os

from argparse import ArgumentParser
from pathlib import Path
from requests import head

from huggingface_hub import snapshot_download

from .convert_vocabulary import convert_vocabulary
from .convert_onnx import convert_model_to_onnx
from .quantize import quantize_model
from .convert_ort import convert_model_to_ort


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Convert an OCR model from disk to ONNX format and then to ORT format.
        The converter is intended to be used with PaddleOCR models, but might work with other models in the future as well.
        This function manages the overall workflow from exporting the model to ONNX and quantizing the model components.
        After the model is quantized, it is converted to ORT format for deployment on ARM devices.
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Provide the HuggingFace model for the pre-trained model to convert."
             "For the moment, only a PaddleOCR model is supported.",
        required=True,
    )

    parser.add_argument(
        "--export_dir",
        type=Path,
        default=Path("export"),
        help="Provide an output directory for the converted model's and configuration file."
             "If unspecified, the converted ORT format model's will be in the '/output' directory, in the provided language.",
    )

    parsed_args = parser.parse_args()
    return parsed_args

def repository_exists(repo_name):
    url = f"https://huggingface.co/{repo_name}"
    response = head(url)
    return response.status_code == 200

def main(
        model: str,
        export_dir: Path,
):
    if repository_exists(model):
        output = snapshot_download(repo_id=model, local_dir_use_symlinks=False)
        model_path = Path(output)
    else:
        raise "The provided model path does not exist."


    output_dir = export_dir / model.split("/")[-1].lower()
    intermediates_dir = output_dir / "intermediates"
    converted_dir = intermediates_dir / "converted"
    quantization_dir = intermediates_dir / "quantized"

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    quantization_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert the model to ONNX format
    convert_model_to_onnx(model_path, converted_dir)

    # Step 2: Quantize the ONNX model
    # quantize_model(converted_dir, "model.onnx", quantization_dir)

    # Step 3: Convert the ONNX model to ORT format
    ort_files = convert_model_to_ort(converted_dir, output_dir)
    vacab_file = convert_vocabulary(output_dir, model_path / "config.json")

if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        export_dir=args.export_dir
    )