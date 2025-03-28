import os

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
from typing import List

from .metadata import load_metadata_for_input_dirs, generate_metadata
from .language import validate_translation_pairs, extract_unique_languages, update_metadata_file
from .bundle_tar import bundle_files
from .utils import copy_folders, remove_folder

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()

def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Bundle multiple translation models into a single tarball file.
        The tarball is directly deployable to the Versta application for translation purposes.
        The provided models should first have been converted using the 'convert' module.
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Provide the directories containing the models to bundle."
        "When translation pairs is enabled, multiple directories should be provided for each language in the pair."
        "For example, if the languages are 'en' and 'nl', provide the directories for 'en-nl' and 'nl-en'.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Provide an output directory for the converted model's and configuration file. "
        "If unspecified, the converted ORT format model's will be in the '/output' directory.",
   )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
        "This will default to False if not specified.",
    )

    parser.add_argument(
        "--keep_input",
        action="store_true",
        default=False,
        help="Whether to remove input file directories after bundeling."
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
    """
    Main function to bundle multiple translation models into a single tarball file.
    The tarball is directly deployable to the Versta application for translation purposes.
    The provided models should first have been converted using the 'convert' module.

    Args:
        input_dir (Path): List of directories containing the models to bundle.
    """
    name = input_dir.name

    # Step 1: Load metadata for the input directories
    metadata = load_metadata_for_input_dirs(input_dir)

    bundle_output_dir = output_dir / f"{name}-bundle"
    intermediates_dir = bundle_output_dir / "intermediates"

    bundle_output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Copy the input directories to the intermediate directory
    copy_folders(input_dir, intermediates_dir)

    # Step 3: Generate metadata for the model conversion process
    generate_metadata(version, intermediates_dir, metadata)

    # Step 4: Bundle the folders into a single .tar.gz file
    output_archive = bundle_output_dir / f"{name}-bundle.tar.gz"
    output_files = list(intermediates_dir.iterdir())

    bundle_files(output_files, output_archive)

    # Step 5: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 9: Remove input directories if specified
    if not keep_input:
        remove_folder(input_dir)
        print(f"Input directories removed.")

if __name__ == "__main__":
    args = parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keep_intermediates=args.keep_intermediates,
        keep_input=args.keep_input,
    )
