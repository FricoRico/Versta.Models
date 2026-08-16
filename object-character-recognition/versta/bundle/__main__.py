import json
import os

from argparse import ArgumentParser
from pathlib import Path
from typing import List, TypedDict

from versta.export.definitions import PACK_NAME
from versta.export.typing import Manifest

from .bundle_tar import bundle_files, create_checksum, sha256_file
from .catalog import update_catalog


class Output(TypedDict):
    bundle: Path
    checksum: Path


with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__).replace(".py", ""),
        description="""Bundle a converted OCR pack into a single tarball file.
        The tarball is directly deployable to the Versta application for OCR purposes.
        The pack should first have been produced using the 'export' module.
        Also refreshes the checked-in models.json catalog entry.
        """,
    )

    parser.add_argument(
        "--unique_id",
        type=str,
        default="paddle-ocr",
        help="Provide the unique identifier for the model in the models.json catalog. "
        "Defaults to 'paddle-ocr'.",
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("output") / PACK_NAME,
        help=f"Provide the pack directory produced by the export module. "
        f"If unspecified, 'output/{PACK_NAME}' is used.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Provide an output directory for the bundle and checksum file. "
        "If unspecified, the bundle will be in the 'output' directory.",
    )

    return parser.parse_args()


def verify_pack(pack_dir: Path) -> List[Path]:
    """
    Verifies a pack directory against its manifest.json: every listed file
    must exist with matching size and SHA256.

    Args:
        pack_dir (Path): The pack directory produced by the export module.

    Returns:
        List[Path]: All files to bundle (manifest plus every listed file).

    Raises:
        FileNotFoundError: If the manifest or a listed file is missing.
        ValueError: If a listed file's size or SHA256 does not match.
    """
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found in {pack_dir}")

    with open(manifest_path, "r") as f:
        manifest: Manifest = json.load(f)

    files = [manifest_path]
    for entry in manifest["files"]:
        file_path = pack_dir / entry["name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Pack file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["sizeBytes"]:
            raise ValueError(
                f"{entry['name']}: size mismatch (actual {size}, manifest {entry['sizeBytes']})"
            )
        digest = sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['name']}: sha256 mismatch")
        files.append(file_path)

    return files


def main(
    unique_id: str,
    input_dir: Path,
    output_dir: Path,
) -> Output:
    """
    Bundles a converted OCR pack into a single tarball file + checksum and
    refreshes the checked-in models.json catalog entry.

    Args:
        unique_id (str): Unique identifier for the model in the catalog.
        input_dir (Path): Pack directory produced by the export module.
        output_dir (Path): Directory where the bundle and checksum are saved.

    Returns:
        Output: A dictionary containing the bundle tarball and checksum paths.
    """
    files = verify_pack(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_file = bundle_files(files, output_dir / f"{unique_id}-bundle.tar.gz")
    checksum_file = create_checksum(bundle_file)
    print(f"Checksum written to {checksum_file}")

    catalog_path = update_catalog(unique_id, version, bundle_file, checksum_file)
    print(f"Catalog updated: {catalog_path}")

    return Output(
        bundle=bundle_file,
        checksum=checksum_file,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        unique_id=args.unique_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
