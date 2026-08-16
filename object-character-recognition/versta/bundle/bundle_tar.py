import tarfile
from hashlib import sha256
from pathlib import Path
from typing import List


def sha256_file(path: Path) -> str:
    """
    Computes the hex SHA256 of a file, streaming in 1 MiB chunks.

    Args:
        path (Path): File to hash.

    Returns:
        str: The lowercase hex digest.
    """
    digest = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_files(files: List[Path], output_file: Path) -> Path:
    """
    Bundles the specified files into a single .tar.gz file, flat at the
    archive root.

    Args:
        files (List[Path]): File paths to be bundled.
        output_file (Path): Path for the output .tar.gz file.

    Returns:
        Path: The written archive path.
    """
    print(f"Bundling files into {output_file}")

    with tarfile.open(output_file, "w:gz") as tar:
        for file in sorted(files, key=lambda f: f.name):
            tar.add(file, arcname=file.name)

    return output_file


def create_checksum(file_path: Path) -> Path:
    """
    Computes the SHA-256 checksum of a file and writes it next to it.

    Args:
        file_path (Path): Path to the file for which the checksum is computed.

    Returns:
        Path: The written checksum file path.
    """
    checksum_filename = file_path.with_suffix(".sha256")

    with open(checksum_filename, "w") as f:
        f.write(sha256_file(file_path))

    return checksum_filename
