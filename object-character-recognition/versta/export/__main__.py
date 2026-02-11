import os

from argparse import ArgumentParser
from pathlib import Path
from requests import head
from huggingface_hub.constants import default_cache_path

from huggingface_hub import snapshot_download

from .convert_tflite import convert_paddle_to_tflite
from .metadata import generate_metadata
from .tokenizer import save_tokenizer
from .utils import remove_folder
from .typing import ModelFiles

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Convert an OCR model from disk to TFLite format.
        The converter is intended to be used with PaddleOCR models, but might work with other models in the future as well.
        This function manages the overall workflow from exporting the model to TFLite format with optional quantization.
        After conversion, the model is ready for deployment on Android devices with NPU acceleration via LiteRT.
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
        "This could be either 'detector' or 'recognizer', defaulting to 'recognizer'.",
    )

    parser.add_argument(
        "--export_dir",
        type=Path,
        default=Path("export"),
        help="Provide an output directory for the converted model's and configuration file."
        "If unspecified, the converted TFLite model will be in the '/export' directory.",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "dynamic_int8", "fp16"],
        default="dynamic_int8",
        help="TFLite quantization type. 'dynamic_int8' (default) provides good balance of speed and accuracy."
        "'fp16' for GPU acceleration. 'none' for full precision (larger size).",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to keep intermediate files created during the conversion process."
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
    module: str = "recognizer",
    quantization: str = "dynamic_int8",
    keep_intermediates: bool = False,
    clear_cache: bool = False,
):
    if repository_exists(model):
        output = snapshot_download(repo_id=model, local_dir_use_symlinks=False)
        model_path = Path(output)
    else:
        raise ValueError("The provided model path does not exist.")

    output_dir = export_dir / model.split("/")[-1].lower()
    intermediates_dir = output_dir / "intermediates"

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting export of {model} to TFLite format...")
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {quantization}")

    # Step 1: Convert Paddle to TFLite
    tflite_path = convert_paddle_to_tflite(
        model_path, output_dir, quantization, intermediates_dir, keep_intermediates
    )

    model_files = ModelFiles(
        model=tflite_path.name,
        quantization=quantization if quantization != "none" else None,
    )

    # Step 2: Save tokenizer for recognizer models
    tokenizer_files = save_tokenizer(model_path, output_dir)

    # Step 3: Validate the presence of vocabulary file based on model format
    if tokenizer_files is None:
        if module == "recognizer":
            raise ValueError("Missing vocabulary file for recognizer model.")
    else:
        if module == "detector":
            raise ValueError("Unexpected vocabulary file for detector model.")

    # Step 4: Create metadata file for the exported model
    generate_metadata(
        version, output_dir, model, module, model_files, tokenizer_files, quantization
    )

    # Step 5: Clear the cache if specified
    if clear_cache:
        remove_folder(Path(default_cache_path) / f"models/{model}".replace("/", "--"))
        print("HuggingFace cache cleaned.")

    print(f"Export complete! Model saved to: {tflite_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        export_dir=args.export_dir,
        module=args.module,
        quantization=args.quantization,
        keep_intermediates=args.keep_intermediates,
        clear_cache=args.clear_cache,
    )
