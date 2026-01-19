import os

from argparse import ArgumentParser
from pathlib import Path
from typing import List, TypedDict

from .metadata import load_metadata_for_input_dirs, generate_metadata
from .language import extract_unique_languages, extract_unique_modules
from .bundle_tar import bundle_files, create_checksum
from .utils import copy_folders, remove_folder


class Output(TypedDict):
    bundle: Path
    checksum: Path


with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Bundle multiple OCR models into a single tarball file.
        The tarball is directly deployable to the Versta application for OCR purposes.
        The provided models should first have been converted using the 'export' module.
        Models can be of mixed module types (detect and/or recognizer).
        """,
    )

    parser.add_argument(
        "--unique_id",
        type=str,
        help="Provide a unique identifier for the model. This will be used to generate the metadata file.",
    )

    parser.add_argument(
        "--input_dir",
        nargs="+",
        type=Path,
        help="Provide the directories containing the OCR models to bundle. "
             "Multiple directories can be provided for different models or module types.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Provide an output directory for the bundled model and configuration file. "
             "If unspecified, the bundle will be in the '/output' directory.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to keep intermediate files created during the bundling process. "
             "This will default to False if not specified.",
    )

    parser.add_argument(
        "--keep_input",
        action="store_true",
        default=False,
        help="Whether to keep input file directories after bundling. "
             "This will default to False if not specified.",
    )

    parsed_args = parser.parse_args()
    return parsed_args


def main(
        unique_id: str,
        input_dirs: List[Path],
        output_dir: Path,
        keep_intermediates: bool = False,
        keep_input: bool = False,
) -> Output:
    """
    Main function to bundle multiple OCR models into a single tarball file.
    The tarball is directly deployable to the Versta application for OCR purposes.
    The provided models should first have been converted using the 'export' module.

    Args:
        input_dirs (List[Path]): List of directories containing the models to bundle.
        output_dir (Path): Directory where the bundled file will be saved.
        keep_intermediates (bool): Whether to keep intermediate files created during the bundling process.
        keep_input (bool): Whether to keep input file directories after bundling.

    Returns:
        Output: A dictionary containing the path to the bundled file and checksum file.
    """
    # Step 1: Load metadata for the input directories
    metadata = load_metadata_for_input_dirs(input_dirs)

    # Step 2: Extract the unique languages from the metadata (wildcard override)
    languages = extract_unique_languages(metadata)

    # Step 3: Extract the unique modules from the metadata
    modules = extract_unique_modules(metadata)

    # Create output directory structure
    bundle_output_dir = output_dir
    modules_name = "-".join(modules)
    intermediates_dir = bundle_output_dir / f"{modules_name}-intermediates"

    bundle_output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Copy the input directories to the intermediate directory
    copy_folders(input_dirs, intermediates_dir)

    # Step 5: Generate metadata for the bundle
    generate_metadata(unique_id, version, intermediates_dir, languages, modules, metadata)

    # Step 6: Bundle the folders into a single .tar.gz file
    output_archive = bundle_output_dir / f"{modules_name}-bundle.tar.gz"
    output_files = list(intermediates_dir.iterdir())

    bundle_file = bundle_files(output_files, output_archive)
    checksum_file = create_checksum(bundle_file)

    # Step 7: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediate files cleaned.")

    # Step 8: Remove input directories if specified
    if not keep_input:
        for input_dir in input_dirs:
            remove_folder(input_dir)

        print(f"Input directories removed.")

    return Output(
        bundle=bundle_file,
        checksum=checksum_file,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        unique_id=args.unique_id,
        input_dirs=args.input_dir,
        output_dir=args.output_dir,
        keep_intermediates=args.keep_intermediates,
        keep_input=args.keep_input,
    )

