from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional

from .convert_mnn import resolve_mnnconvert
from .definitions import MODELS, PACK_NAME
from .manifest import write_manifest
from .pipeline import convert_model
from .typing import ManifestFile
from .utils import remove_folder

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="""Download the official PaddleOCR/PaddleClas inference models and convert them
        to MNN int8, producing the PP-OCRv6 OCR pack consumed by the Versta app.

        Pipeline per model: download tar -> extract the Paddle inference model (PIR or legacy
        pdmodel format) -> paddle2onnx (opset 14 for PP-OCRv6, opset 11 for PULC) -> MNNConvert
        with int8 weight quantization. Detector variants with the DBNet head deconvs folded to
        1/2 and 1/4 output resolution are produced as well; recognizer tiers ship their own
        character dictionaries (the tiny tier drops Japanese kana).
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Output directory; the pack lands in <output_dir>/paddle-ocr-v6/.",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        choices=[m["stem"] for m in MODELS],
        help="Convert only these models (by tar stem). Defaults to all.",
    )

    parser.add_argument(
        "--mnnconvert",
        type=Path,
        default=None,
        help="Path to a prebuilt MNNConvert binary. If omitted, the vendored MNN submodule"
        " is built on first use.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Keep downloaded tars, extracted models and ONNX intermediates.",
    )

    parser.add_argument(
        "--glyphmatte_onnx",
        type=Path,
        default=None,
        help="Local glyphmatte ONNX override: skips the HF download (use before the "
        "model is published or to exercise a fresh training pipeline output).",
    )

    return parser.parse_args()


def main(
    output_dir: Path,
    models: Optional[List[str]] = None,
    mnnconvert: Optional[Path] = None,
    keep_intermediates: bool = False,
    glyphmatte_onnx: Optional[Path] = None,
) -> Path:
    selected = [m for m in MODELS if models is None or m["stem"] in models]
    if glyphmatte_onnx is not None:
        for spec in selected:
            if spec["stem"] == "glyphmatte":
                spec.pop("hf", None)
                spec["url"] = glyphmatte_onnx.resolve().as_uri()

    output_dir = output_dir / PACK_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    converter = resolve_mnnconvert(mnnconvert)

    print("Downloading required source files...")
    entries: List[ManifestFile] = []
    for spec in selected:
        entries.extend(convert_model(spec, intermediates_dir, output_dir, converter))

    manifest_path = write_manifest(version, output_dir, entries)
    print(manifest_path)

    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediate files cleaned.")

    return output_dir


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        models=args.models,
        mnnconvert=args.mnnconvert,
        keep_intermediates=args.keep_intermediates,
        glyphmatte_onnx=args.glyphmatte_onnx,
    )
