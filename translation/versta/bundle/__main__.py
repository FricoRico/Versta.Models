import json
import os

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import List, TypedDict, Union

from ..leanmt import assert_leanmt_compatible
from ..schema import (
    validate_bundle_archive,
    validate_bundle_metadata,
    validate_direction_metadata,
)
from .metadata import load_metadata_for_input_dirs, generate_metadata
from .language import (
    assert_boundary_codes,
    validate_translation_pairs,
    extract_unique_languages,
)
from .bundle_tar import bundle_files, create_checksum
from .utils import copy_folders, remove_folder


class Output(TypedDict):
    bundle: Path
    checksum: Path


with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def parse_args() -> Namespace:
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Bundle multiple Firefox (Bergamot) translation models into a single tarball file.
        The tarball is directly deployable to the Versta application for translation purposes.
        The provided models should first have been downloaded using the 'download' module.
        """,
    )

    parser.add_argument(
        "--input_dir",
        nargs="+",
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
        "--bidirectional",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether the languages are a bidirectional pair, e.g. 'en-nl' and 'nl-en'."
        "A language pair allows for the translation model to be used in both directions easily."
        "This will default to True if not specified.",
    )

    parser.add_argument(
        "--subdirectory",
        type=bool,
        default=False,
        help="Whether the output should be moved out of the subdirectory."
        "It will temporarily place the files in a subdirectory and then move them out if subdirectory is False."
        "This will default to False if not specified.",
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
    input_dirs: List[Path],
    output_dir: Path,
    bidirectional: bool = True,
    subdirectory: bool = False,
    keep_intermediates: bool = False,
    keep_input: bool = False,
) -> Output:
    """
    Main function to bundle multiple Firefox (Bergamot) translation models into a single tarball file.
    The tarball is directly deployable to the Versta application for translation purposes.
    The provided models should first have been downloaded using the 'download' module.

    Args:
        input_dirs (list[Path]): List of directories containing the models to bundle.
        output_dir (Path): Directory where the bundled file will be saved.
        bidirectional (bool): Whether the languages are a bidirectional pair, e.g. 'en-nl' and 'nl-en'.
        keep_intermediates (bool): Whether to remove intermediate files created during the bundling process.
        keep_input (bool): Whether to remove input file directories after bundling.

    Returns:
        (Output): A dictionary containing the path to the bundled file and checksum file.
    """
    # Step 0: Fail-fast validation of every input direction against the app's
    # per-direction loader contract, including file existence on disk, plus the
    # fail-closed leanmt handshake on each model payload (NEU-25).
    for input_dir in input_dirs:
        with open(input_dir / "metadata.json", "r", encoding="utf-8") as handle:
            direction_metadata = json.load(handle)
        validate_direction_metadata(direction_metadata, base_dir=input_dir)
        assert_leanmt_compatible(input_dir / str(direction_metadata["files"]["model"]))

    # Step 1: Load metadata for the input directories
    metadata = load_metadata_for_input_dirs(input_dirs)

    # Step 2: Validate the translation pairs, and refuse non-canonical
    # language codes at the bundle boundary (NEU-133): bundle manifests and
    # archive names carry registry-normalized codes only.
    assert_boundary_codes(metadata)
    if bidirectional:
        validate_translation_pairs(metadata)

    # Step 3: Extract the unique languages from the metadata
    languages = extract_unique_languages(metadata)

    if not subdirectory:
        bundle_output_dir = output_dir
        intermediates_dir = bundle_output_dir / f"{'-'.join(languages)}-intermediates"
    else:
        bundle_output_dir = output_dir / f"{'-'.join(languages)}-bundle"
        intermediates_dir = bundle_output_dir / "intermediates"

    bundle_output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Copy the input directories to the intermediate directory
    copy_folders(input_dirs, intermediates_dir)

    # Step 5: Generate metadata for the model conversion process
    metadata_file = generate_metadata(
        version, intermediates_dir, languages, metadata, bidirectional
    )
    with open(metadata_file, "r", encoding="utf-8") as handle:
        validate_bundle_metadata(json.load(handle))

    # Step 6: Bundle the folders into a single .tar.gz file
    output_archive = bundle_output_dir / f"{'-'.join(languages)}-bundle.tar.gz"
    output_files = sorted(intermediates_dir.iterdir())

    bundle_file = bundle_files(output_files, output_archive)
    checksum_file = create_checksum(bundle_file)

    # Step 6b: Fail-fast validation of the produced archive against the app's
    # bundle loader contract before anything is cleaned up or shipped.
    validate_bundle_archive(bundle_file)

    # Step 7: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 8: Remove input directories if specified
    if not keep_input:
        for input_dir in input_dirs:
            remove_folder(input_dir)

        print("Input directories removed.")

    return Output(
        bundle=bundle_file,
        checksum=checksum_file,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dirs=args.input_dir,
        output_dir=args.output_dir,
        bidirectional=args.bidirectional,
        subdirectory=args.subdirectory,
        keep_intermediates=args.keep_intermediates,
        keep_input=args.keep_input,
    )
