import tarfile
from hashlib import sha256

from pathlib import Path
from typing import List

def bundle_files(files: List[str], output_file: Path) -> Path:
    """
    Bundles the specified folders into a single .tar.gz file.

    Args:
        folders (list[Path]): List of folder paths to be bundled.
        output_file (Path): Path for the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        for file in files:
            tar.add(file, arcname=file.name)

    return output_file

def create_checksum(file_path: Path) -> Path:
    """
    Compute the SHA-256 checksum of a file and save it to a file.

    Args:
        file_path (str): The path to the file for which the checksum is to be computed.

    Returns:
        str: The computed SHA-256 checksum as a hexadecimal string.
    """
    hash = sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash.update(byte_block)

    checksum = hash.hexdigest()
    checksum_filename = file_path.with_suffix(".sha256")

    with open(checksum_filename, "w") as f:
        f.write(checksum)

    return checksum_filename