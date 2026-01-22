import os

from argparse import ArgumentParser
from pathlib import Path
from requests import head
from huggingface_hub.constants import default_cache_path

from huggingface_hub import snapshot_download

from .convert_onnx import convert_model_to_onnx
from .optimize_onnx import optimize_model
from .convert_ort import convert_model_to_ort
from .metadata import generate_metadata
from .tokenizer import save_tokenizer
from .utils import remove_folder
from .quantize_dynamic_q8 import quantize_model_to_q8_fp16_dynamic

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Convert an OCR model from disk to ONNX format and then to ORT format.
        The converter is intended to be used with PaddleOCR models, but might work with other models in the future.
        This function manages overall workflow from exporting model to ONNX, applying quantization (optional),
        and converting to ORT format for deployment on ARM devices.

        Pipeline (default - FP16 only):
        1. Export to FP16 ONNX
        2. Optimize with ONNX-Slim (single pass)
        3. Convert to ORT format

        Pipeline (with QInt8 quantization):
        1. Export to FP32 ONNX (DynamicQuantizeLinear requires FP32)
        2. Quantize to QInt8 weights with QInt16 (FP16-like) activations (dynamic quantization)
        3. Convert to ORT format (skip ONNX-Slim - incompatible with quantization)
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
        "This could be either 'detector' or 'recognizer', defaulting to 'detector'.",
    )

    parser.add_argument(
        "--export_dir",
        type=Path,
        default=Path("export"),
        help="Provide an output directory for the converted model's and configuration file."
        "If unspecified, the converted ORT format model's will be in the '/output' directory, in the provided language.",
    )

    parser.add_argument(
        "--prune_ratio",
        type=float,
        default=0.0,
        help="Fraction of channels to prune (0.0 to 0.5). "
        "Use PaddleSlim to reduce model size and improve speed. "
        "Note: PaddleSlim pruning is not supported for inference models. "
        "Pruning must be done during training. Default: 0.0 (no pruning).",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Enable single ONNX-Slim pass after quantization. "
        "Recommmended: True to clean up quantization artifacts before ORT conversion. "
        "Default: True.",
    )

    parser.add_argument(
        "--no_optimize",
        action="store_true",
        default=False,
        help="Disable ONNX-Slim optimization. "
        "Useful for debugging quantization issues. "
        "Default: False.",
    )

    parser.add_argument(
        "--precision",
        choices=["fp16", "qint8_fp16"],
        default="fp16",
        help="Precision for exported model. "
        "'fp16': FP16 weights and activations (baseline, current ~1.6s/doc) "
        "'qint8_fp16': QInt8 weights with QInt16 (FP16-like) activations using dynamic quantization "
        "(no calibration needed, MatMul, Conv, Gemm, Add, Mul operators). "
        "Recommmended: 'qint8_fp16' for faster inference (~40-50% speedup). "
        "Default: 'fp16'.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during conversion process."
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
    prune_ratio: float = 0.0,
    optimize: bool = True,
    no_optimize: bool = False,
    precision: str = "fp16",
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
    converted_dir = intermediates_dir / "converted"
    quantization_dir = intermediates_dir / "quantized"
    optimized_dir = intermediates_dir / "optimized"

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    quantization_dir.mkdir(parents=True, exist_ok=True)
    optimized_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert the model to ONNX format with NO optimization yet
    # Use FP32 for quantization (DynamicQuantizeLinear requires FP32), FP16 for baseline
    export_precision = "fp32" if precision == "qint8_fp16" else "fp16"
    onnx_model_path = convert_model_to_onnx(
        model_path=model_path,
        export_dir=converted_dir,
        prune_ratio=prune_ratio,
        precision=export_precision,
    )

    # Step 2: Apply QInt8 quantization if requested (no optimization yet)
    final_model = onnx_model_path
    intermediate_dir = converted_dir

    if precision == "qint8_fp16":
        quantization_dir.mkdir(parents=True, exist_ok=True)
        quantized_model_path = quantization_dir / "model.onnx"

        print(f"\n{'=' * 70}")
        print(f"Applying QInt8 FP16 dynamic quantization")
        print(f"{'=' * 70}")
        print(f"Using dynamic quantization (no calibration)")
        print(f"INT8 weights + QInt16 (FP16-like) activations")
        print(f"Multiple operators: MatMul, Conv, Gemm, Add, Mul")
        print(f"{'=' * 70}\n")

        final_model = quantize_model_to_q8_fp16_dynamic(
            model_path=onnx_model_path,
            output_path=quantized_model_path,
        )
        intermediate_dir = quantization_dir

        print(f"\n✓ QInt8 quantization complete: {final_model}")

    # Step 3: Single ONNX-Slim pass AFTER quantization, BEFORE ORT conversion
    # Only optimize for FP16 baseline - skip for QInt8 to avoid graph corruption
    if not no_optimize and optimize and precision == "fp16":
        print(f"\n{'=' * 70}")
        print(f"Running ONNX-Slim optimization")
        print(f"{'=' * 70}")
        print(f"Single pass for FP16 baseline")
        print(f"{'=' * 70}\n")

        optimized_path = intermediate_dir / "optimized.onnx"
        final_model = optimize_model(
            model_path=final_model,
            output_path=optimized_path,
            num_passes=1,
        )
        print(f"✓ Optimization complete: {final_model}")
    elif precision == "qint8_fp16":
        print(f"\n⏭️  Skipping ONNX-Slim for QInt8 quantization")
        print(f"   (DynamicQuantizeLinear doesn't work well with graph optimization)")
    elif no_optimize:
        print(f"\n⏭️  Optimization disabled (no_optimize=True)")

    # Step 4: Convert to ORT format
    if precision == "qint8_fp16":
        ort_files = convert_model_to_ort(quantization_dir, output_dir)
    elif precision == "fp16" and optimize and not no_optimize:
        ort_files = convert_model_to_ort(optimized_dir, output_dir)
    else:
        ort_files = convert_model_to_ort(converted_dir, output_dir)
    tokenizer_files = save_tokenizer(model_path, output_dir)

    # Step 5: Validate the presence of vocabulary file based on model format
    if tokenizer_files is None:
        if module == "recognizer":
            raise ValueError("Missing vocabulary file for recognizer model.")
    else:
        if module == "detector":
            raise ValueError("Unexpected vocabulary file for detector model.")

    # Step 6: Create metadata file for the exported model
    generate_metadata(version, output_dir, model, module, ort_files, tokenizer_files)

    # Step 7: Clean up intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 8: Clear the cache if specified
    if clear_cache:
        remove_folder(Path(default_cache_path) / f"models/{model}".replace("/", "--"))
        print("HuggingFace cache cleaned.")


if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        export_dir=args.export_dir,
        module=args.module,
        prune_ratio=args.prune_ratio,
        optimize=args.optimize,
        no_optimize=args.no_optimize,
        precision=args.precision,
        keep_intermediates=args.keep_intermediates,
        clear_cache=args.clear_cache,
    )
