import os

from argparse import ArgumentParser
from pathlib import Path
from requests import head
from huggingface_hub.constants import default_cache_path

from huggingface_hub import snapshot_download

from .convert_onnx import convert_model_to_onnx
from .convert_ort import convert_model_to_ort
from .metadata import generate_metadata
from .tokenizer import save_tokenizer
from .utils import remove_folder

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()

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
        "--module",
        type=str,
        default="recognizer",
        help="Specify the format of the model to convert."
             "This could be either 'detector' or 'recognizer', defaulting to 'detector'."
    )

    parser.add_argument(
        "--export_dir",
        type=Path,
        default=Path("export"),
        help="Provide an output directory for the converted model's and configuration file."
             "If unspecified, the converted ORT format model's will be in the '/output' directory, in the provided language.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
             "This will default to False if not specified.",
    )

    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=False,
        help="Whether to remove the downloaded files from HuggingFace cache."
             "This will default to False if not specified.",
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
        module: str = "detector",
        keep_intermediates: bool = False,
        clear_cache: bool = False,
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

    # Step 2: Convert the ONNX model to ORT format
    ort_files = convert_model_to_ort(converted_dir, output_dir)
    tokenizer_files = save_tokenizer(model_path, output_dir)

    # Step 3: Validate the presence of vocabulary file based on model format
    if tokenizer_files is None:
        if module == "recognizer":
            raise ValueError("Missing vocabulary file for recognizer model.")
    else:
        if module == "detector":
            raise ValueError("Unexpected vocabulary file for detector model.")

    # Step 4: Create metadata file for the exported model
    generate_metadata(version, output_dir, model, module, ort_files, tokenizer_files)

    # Step 5: Clean up intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 6: Clear the cache if specified
    if clear_cache:
        remove_folder(Path(default_cache_path) / f"models/{model}".replace("/", "--"))
        print("HuggingFace cache cleaned.")

if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        export_dir=args.export_dir,
        module=args.module,
        keep_intermediates=args.keep_intermediates,
        clear_cache=args.clear_cache,
    )