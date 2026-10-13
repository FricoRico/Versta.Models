import gzip
import os
import tarfile

from hashlib import sha256
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# Canonical modification time stamped on every archive member. A fixed epoch makes
# the produced tarball byte-identical for identical inputs; SOURCE_DATE_EPOCH
# overrides it when an external builder wants to pin a release timestamp.
DEFAULT_MTIME = 0

_CANONICAL_FILE_MODE = 0o644
_CANONICAL_DIR_MODE = 0o755


def _canonical_mtime() -> int:
    """
    Resolve the mtime to stamp on all archive members.

    Returns:
        int: SOURCE_DATE_EPOCH when set, otherwise the fixed DEFAULT_MTIME.
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        try:
            return int(epoch)
        except ValueError:
            raise ValueError(
                f"SOURCE_DATE_EPOCH must be an integer epoch, got: {epoch!r}"
            )
    return DEFAULT_MTIME


def _normalize_tarinfo(info: tarfile.TarInfo, mtime: int) -> tarfile.TarInfo:
    """
    Strip all host- and run-dependent attributes from an archive member so the
    output depends only on the member's relative path, type and contents.

    Args:
        info (tarfile.TarInfo): The member description to normalize in place.
        mtime (int): Canonical modification time to stamp.

    Returns:
        tarfile.TarInfo: The normalized member description.
    """
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""

    if info.isdir():
        info.mode = _CANONICAL_DIR_MODE
    else:
        info.mode = _CANONICAL_FILE_MODE

    return info


def _iter_members(
    path: Path, arcname: str, mtime: int
) -> Iterator[Tuple[tarfile.TarInfo, Optional[Path]]]:
    """
    Yield normalized TarInfo members for `path` and, for directories, every
    entry below it, ordered deterministically by archive path. Directories are
    yielded before their contents and file data is read from disk.

    Args:
        path (Path): File or directory to include in the archive.
        arcname (str): Archive path of the top-level member.
        mtime (int): Canonical modification time stamped on every member.

    Yields:
        tuple[tarfile.TarInfo, Path | None]: The member description and the
        on-disk file to read for regular files (None for directories).
    """
    info = tarfile.TarInfo(arcname)

    if path.is_dir():
        info.type = tarfile.DIRTYPE
        yield _normalize_tarinfo(info, mtime), None
        for child in sorted(path.iterdir(), key=lambda entry: entry.name):
            yield from _iter_members(child, f"{arcname}/{child.name}", mtime)
    else:
        info.type = tarfile.REGTYPE
        info.size = path.stat().st_size
        yield _normalize_tarinfo(info, mtime), path


def bundle_files(files: List[Path], output_file: Path) -> Path:
    """
    Bundles the specified folders into a single .tar.gz file.

    The archive is deterministic: members are ordered by archive path and every
    host-dependent attribute (mtime, uid/gid, uname/gname, permissions) is
    normalized, so identical inputs produce byte-identical output.

    Args:
        files (list[Path]): List of folder paths to be bundled.
        output_file (Path): Path for the output .tar.gz file.
    """
    print(f"Bundling files into {output_file}")

    mtime = _canonical_mtime()
    members = sorted(files, key=lambda entry: entry.name)

    with open(output_file, "wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=mtime
        ) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w") as tar:
                for member in members:
                    for info, source in _iter_members(member, member.name, mtime):
                        if source is None:
                            tar.addfile(info)
                        else:
                            with open(source, "rb") as handle:
                                tar.addfile(info, handle)

    return output_file


def create_checksum(file_path: Path) -> Path:
    """
    Compute the SHA-256 checksum of a file and save it to a file.

    Args:
        file_path (str): The path to the file for which the checksum is to be computed.

    Returns:
        Path: The path of the written checksum file.
    """
    print(f"Creating checksum for {file_path}")

    hash = sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash.update(byte_block)

    checksum = hash.hexdigest()
    checksum_filename = file_path.with_suffix(".sha256")

    with open(checksum_filename, "w") as f:
        f.write(checksum)

    return checksum_filename
