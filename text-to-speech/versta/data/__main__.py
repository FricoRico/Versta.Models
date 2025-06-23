import os

from argparse import ArgumentParser
from pathlib import Path

from .download import download_folder_from_git, download_folder_from_tarball
from .metadata import generate_metadata
from .utils import remove_folder
from .espeak import build_data

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
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Provide an output directory for the converted model's and configuration file. "
        "If unspecified, the converted ORT format model's will be in the '/output' directory.",
   )

    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=Path("tmp"),
        help="Provide an temporary directory used to download and extract raw models."
             "If unspecified, the downloaded models will go into '/tmp' directory.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
        "This will default to False if not specified.",
    )

    parser.add_argument(
        "--keep_downloads",
        action="store_true",
        default=False,
        help="Whether to remove downloaded files created during the conversion process."
        "This will default to False if not specified.",
    )

    parsed_args = parser.parse_args()
    return parsed_args

def main(
    output_dir: Path,
    temp_dir: Path,
    keep_intermediates: bool = False,
    keep_downloads: bool = False,
):
    """
    Main function to bundle multiple translation models into a single tarball file.
    The tarball is directly deployable to the Versta application for translation purposes.
    The provided models should first have been converted using the 'convert' module.

    Args:
        output_dir (Path): The directory where the output tarball will be saved.
        temp_dir (Path): The directory where temporary files will be stored.
        keep_intermediates (bool, optional): Whether to keep intermediate files. Defaults to False.
        keep_downloads (bool, optional): Whether to keep downloaded files. Defaults to False.
    """
    name = "versta-tts-data"

    export_dir = output_dir / name

    download_dir = temp_dir / "downloads"
    intermediates_dir = output_dir / name / "intermediates"
    build_dir = intermediates_dir / "builds"

    download_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download the data folders
    espeak_path = download_folder_from_git(export_path=build_dir, download_path=download_dir, repo_url="git@github.com:espeak-ng/espeak-ng.git")
    espeak_data_path = build_data(espeak_path, build_dir, export_dir / "espeak-ng-data")
    open_jtalk_data_path = download_folder_from_tarball(export_path=export_dir, download_path=download_dir, tarball_url="http://downloads.sourceforge.net/open-jtalk/open_jtalk_dic_utf_8-1.11.tar.gz", folder_path="open-jtalk-data")

    # Step 3: Generate metadata for the model conversion process
    generate_metadata(name, version, export_dir, espeak_data_path, open_jtalk_data_path)

    # Step 4: Remove intermediate files if specified
    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")

    # Step 5: Remove download files if specified
    if not keep_downloads:
        remove_folder(download_dir)
        print("Downloads files cleaned.")

if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        keep_intermediates=args.keep_intermediates,
        keep_downloads=args.keep_downloads,
    )
