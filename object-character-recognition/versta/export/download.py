import tarfile

from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm


def download_tar(url: str, dest: Path) -> Path:
    """
    Downloads a model tar archive, streaming to disk with a progress bar when
    the server reports a content length.

    Args:
        url (str): The upstream tar URL.
        dest (Path): Destination file path.

    Returns:
        Path: The downloaded tar path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, method="GET"), timeout=600) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with open(dest, "wb") as out:
            with tqdm(
                desc=dest.name,
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    bar.update(len(chunk))
    return dest


def extract_tar(tar_path: Path, dest_dir: Path) -> Path:
    """
    Extracts a model tar into the destination directory.

    Args:
        tar_path (Path): The tar archive to extract.
        dest_dir (Path): The directory to extract into.

    Returns:
        Path: The extraction directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir, filter="data")
    return dest_dir
