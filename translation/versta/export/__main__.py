import os

from argparse import ArgumentParser
from pathlib import Path
from typing import TypedDict

from huggingface_hub.constants import default_cache_path

from .config import (
    get_source_language,
    get_target_language,
    get_architecture,
    get_architecture_config,
)
from .tokenizer import save_tokenizer, optimize_vocabulary
from .convert_fp16 import convert_models_to_fp16
from .convert_onnx import convert_model_to_onnx
from .convert_ort import convert_model_to_ort
from .metadata import generate_metadata
from .minimize import minimize
from .utils import remove_folder


class Output(TypedDict):
    path: Path
    metadata: Path


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
        default=Path("output"),
        help="Provide an output directory for the converted model's and configuration file."
        "If unspecified, the converted ORT format model's will be in the '/output' directory, in the provided language.",
    )

    parser.add_argument(
        "--score",
        type=float,
        default=0.0,
        help="Provide the score of the model to be converted. "
        "This score will be used to determine the quality of the model and will be included in the metadata file.",
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
    score: float = 0.0,
    keep_intermediates: bool = False,
    clear_cache: bool = False,
) -> Output:
    """
    Main function to handle the model exporting, tokenization, and quantization process.
    This function manages the overall workflow from exporting the model to ONNX, saving the tokenizer,
    and quantizing the model components. After the model is quantized, it is converted to ORT format.

    Args:
        model (str): The name of the pre-trained model to convert.
        output_dir (Path): The output directory for the converted model and configuration file.
        score (float): The score of the model to be converted. This will be included in the metadata file.
        keep_intermediates (bool): Whether to remove intermediate files created during the conversion process.
        clear_cache (bool): Whether to remove the downloaded files from HuggingFace cache.

    Returns:
        Output: A dictionary containing the path to the exported model and metadata file.
    """
    source_language = get_source_language(model)
    target_language = get_target_language(model)
    architectures = get_architecture(model)
    arch_config = get_architecture_config(model)

    language_output_dir = output_dir / f"{source_language}-{target_language}"

    intermediates_dir = language_output_dir / "intermediates"
    converted_dir = intermediates_dir / "converted"
    quantization_dir = intermediates_dir / "quantized"

    language_output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    quantization_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert the model to ONNX format
    convert_model_to_onnx(model, converted_dir)

    # Remove unused decoder_model.onnx (we only use decoder_with_past_model.onnx)
    decoder_model_path = converted_dir / "decoder_model.onnx"
    if decoder_model_path.exists():
        decoder_model_path.unlink()
        print("Removed unused decoder_model.onnx")

    # Step 2: Save the tokenizer and optimize the vocabulary
    tokenizer_files = save_tokenizer(model, language_output_dir)
    tokenizer_files_optimized = optimize_vocabulary(
        tokenizer_files, language_output_dir
    )

    # Step 3: Convert models to FP16 format
    encoder_fp16, decoder_fp16 = convert_models_to_fp16(converted_dir, quantization_dir)

    # Step 4: Convert the FP16 models to ORT format
    ort_files = convert_model_to_ort(quantization_dir, language_output_dir)

    # Step 5: Create metadata file for the model
    generate_metadata(
        version,
        language_output_dir,
        model,
        source_language,
        target_language,
        architectures,
        arch_config,
        score,
        tokenizer_files_optimized,
        ort_files,
    )

    # Step 6: Remove unused files
    minimize(language_output_dir)

    # Step 7: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 8: Clear the cache if specified
    if clear_cache:
        remove_folder(Path(default_cache_path) / f"models/{model}".replace("/", "--"))
        print("HuggingFace cache cleaned.")

    return Output(
        path=language_output_dir,
        metadata=language_output_dir / "metadata.json",
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        output_dir=args.output_dir,
        score=args.score,
        keep_intermediates=args.keep_intermediates,
        clear_cache=args.clear_cache,
    )
