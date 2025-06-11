import os

from argparse import ArgumentParser
from pathlib import Path
from huggingface_hub.constants import default_cache_path

from .quantize import quantize_model
from .convert_onnx import convert_model_to_onnx
from .convert_ort import convert_model_to_ort
from .tokenizer import save_tokenizer
from .metadata import generate_metadata, get_voices
from .utils import remove_folder

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()

def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Convert a model from HuggingFace model hub to ONNX format and then to ORT format.
        The converter is intended to be used with Helsinki-NLP's Opus-MT translation models, but might work with other models as well.
        This function manages the overall workflow from exporting the model to ONNX, saving the tokenizer, and quantizing the model components.
        After the model is quantized, it is converted to ORT format for deployment on ARM devices.
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Provide the name of the pre-trained model to convert."
             "For the moment, only Helsinki-NLP's Opus-MT translation models are supported.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/kokoro"),
        help="Provide an output directory for the converted model's and configuration file."
             "If unspecified, the converted ORT format model's will be in the '/output' directory, in the provided language.",
    )

    parser.add_argument(
        "--model_format",
        type=str,
        default="kokoro",
        help="Specify the format of the model to convert."
             "This could be either 'kokoro' or 'piper' at the moment, defaulting to 'kokoro'."
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

def main(
        model: str,
        output_dir: Path,
        keep_intermediates: bool = False,
        clear_cache: bool = False,
        model_format: str = "kokoro",
):
    print("Exporting the model...")

    intermediates_dir = output_dir / "intermediates"
    converted_dir = intermediates_dir / "converted"
    quantization_dir = intermediates_dir / "quantized"

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    quantization_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert the model to ONNX format
    convert_model_to_onnx(model, converted_dir, model_format)

    # Step 2: Quantize the model
    quantize_model(converted_dir, "model.onnx", quantization_dir, model_format)

    # Step 3: Convert the quantized models to ORT format
    ort_files = convert_model_to_ort(quantization_dir, output_dir)

    # Step 4: Save the tokenizer files
    tokenizer_files = save_tokenizer(converted_dir, output_dir)

    # Step 5: Get all voices from voices directory
    voices = get_voices(output_dir / "voices")

    # Step 6: Create metadata file for the model
    generate_metadata(version, output_dir, model, ["Kokoro"], ort_files, tokenizer_files, voices)

    # Step 6: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 7: Clear the cache if specified
    if clear_cache:
        remove_folder(Path(default_cache_path) / f"models/{model}".replace("/", "--"))
        print("HuggingFace cache cleaned.")

if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        output_dir=args.output_dir,
        keep_intermediates=args.keep_intermediates,
        clear_cache=args.clear_cache,
        model_format=args.model_format,
    )