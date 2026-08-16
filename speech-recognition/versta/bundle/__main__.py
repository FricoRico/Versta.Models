import json
import os

from argparse import ArgumentParser
from pathlib import Path
from typing import TypedDict

from .bundle_tar import bundle_files, create_checksum
from .metadata import generate_bundle_metadata
from .utils import copy_contents, remove_folder


class Output(TypedDict):
    bundle: Path
    checksum: Path


def bundle_id(model_id: str) -> str:
    return f"whisper.{model_id.split('-')[0]}"


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Bundle a previously downloaded whisper.cpp model into a single tarball file.
        The tarball is directly deployable to the Versta application for speech recognition.
        The model should first have been downloaded using the 'export' module.
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing the downloaded model, VAD model and metadata.json. "
        "The model id is read from metadata.json.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Provide an output directory for the bundle and configuration file.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the bundling process."
        "This will default to False if not specified.",
    )

    parser.add_argument(
        "--keep_input",
        action="store_true",
        default=False,
        help="Whether to remove the input directory after bundling."
        "This will default to False if not specified.",
    )

    parsed_args = parser.parse_args()
    return parsed_args


def main(
    input_dir: Path,
    output_dir: Path,
    keep_intermediates: bool = False,
    keep_input: bool = False,
):
    # Step 1: Load the per-model metadata written by the export module. The model id is
    # read from this file, so the bundle step does not need it passed in explicitly.
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}.")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        model_metadata = json.load(handle)

    model_id = model_metadata.get("id") or input_dir.name
    name = bundle_id(model_id)

    intermediates_dir = output_dir / "intermediates"

    intermediates_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Copy the model folder contents into a subfolder named after the model id,
    # so the tarball layout matches the ``directory`` referenced by the bundle metadata.
    copy_contents(input_dir, intermediates_dir / model_id)

    # Step 3: Generate the bundle-level metadata (SpeechRecognitionBundleMetadata schema).
    # Languages are taken automatically from the bundled model metadata.
    generate_bundle_metadata(
        name, model_metadata, intermediates_dir, directory=model_id
    )

    # Step 4: Bundle the folders into a single .tar.gz file
    output_archive = output_dir / f"{name}-bundle.tar.gz"
    output_files = list(intermediates_dir.iterdir())

    bundle_file = bundle_files(output_files, output_archive)
    checksum_file = create_checksum(bundle_file)

    # Step 5: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 6: Remove input directories if specified
    if not keep_input:
        remove_folder(input_dir)
        print("Input directories removed.")

    return Output(
        bundle=bundle_file,
        checksum=checksum_file,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keep_intermediates=args.keep_intermediates,
        keep_input=args.keep_input,
    )
