"""
Deterministic packing and safe extraction of corpus snapshot artifacts.

Packing follows the same normalization rules as `versta.bundle`: members are
ordered by archive path, uid/gid are zeroed, names blanked, modes normalized
and mtimes pinned, so identical inputs produce byte-identical `.tar.gz`.
"""

import gzip
import io
import os
import tarfile

from hashlib import sha256
from pathlib import Path
from typing import Dict, List

# Canonical modification time stamped on every archive member; overridable via
# SOURCE_DATE_EPOCH when an external builder wants to pin a release timestamp.
DEFAULT_MTIME = 0

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hex digest.
    """
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_mtime() -> int:
    """
    Resolve the mtime stamped on all archive members.

    Returns:
        int: SOURCE_DATE_EPOCH when set, otherwise the fixed DEFAULT_MTIME.

    Raises:
        ValueError: If SOURCE_DATE_EPOCH is set but not an integer.
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


def pack_snapshot(payloads: Dict[str, bytes], root: str, output_file: Path) -> Path:
    """
    Pack in-memory payload files into a deterministic `.tar.gz`.

    Args:
        payloads (dict[str, bytes]): Archive path (relative to `root`) to file
            contents. Members are written sorted by archive path.
        root (str): Name of the single top-level directory in the archive.
        output_file (Path): Destination `.tar.gz` path.

    Returns:
        Path: `output_file`, for chaining.
    """
    print(f"Packing corpus snapshot into {output_file}")

    mtime = _canonical_mtime()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=mtime
        ) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w") as tar:
                root_info = tarfile.TarInfo(root)
                root_info.type = tarfile.DIRTYPE
                _normalize(root_info, mtime)
                tar.addfile(root_info)

                dirs = sorted(
                    {name.rsplit("/", 1)[0] for name in payloads if "/" in name}
                )
                for directory in dirs:
                    info = tarfile.TarInfo(f"{root}/{directory}")
                    info.type = tarfile.DIRTYPE
                    _normalize(info, mtime)
                    tar.addfile(info)

                for name in sorted(payloads):
                    data = payloads[name]
                    info = tarfile.TarInfo(f"{root}/{name}")
                    info.type = tarfile.REGTYPE
                    info.size = len(data)
                    _normalize(info, mtime)
                    with io.BytesIO(data) as source:
                        tar.addfile(info, source)

    return output_file


def _normalize(info: tarfile.TarInfo, mtime: int) -> None:
    """
    Strip host- and run-dependent attributes from an archive member in place.

    Args:
        info (tarfile.TarInfo): Member description to normalize.
        mtime (int): Canonical modification time to stamp.
    """
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644


def read_tar_gz_members(archive: Path, prefix: str) -> Dict[str, bytes]:
    """
    Read the regular files under `prefix` of a `.tar.gz` into memory.

    Args:
        archive (Path): Archive to read.
        prefix (str): Member path prefix to select (e.g. "flores200_dataset/dev/").

    Returns:
        dict[str, bytes]: Archive path with `prefix` stripped -> file bytes.

    Raises:
        ValueError: If the archive cannot be read or matches nothing.
    """
    members: Dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        for info in tar:
            name = info.name
            if name.startswith("./"):
                name = name[2:]
            if not info.isfile() or not name.startswith(prefix):
                continue
            handle = tar.extractfile(info)
            if handle is None:
                continue
            members[name[len(prefix) :]] = handle.read()

    if not members:
        raise ValueError(f"{archive} contains no files under '{prefix}'")
    return members


def extract_snapshot(artifact: Path, output_dir: Path) -> Path:
    """
    Extract a snapshot artifact into `output_dir`, refusing unsafe members.

    Args:
        artifact (Path): Snapshot `.tar.gz` produced by `pack_snapshot`.
        output_dir (Path): Directory the snapshot root is extracted into.

    Returns:
        Path: The extracted snapshot root directory.
    """
    roots: List[str] = []
    with tarfile.open(artifact, mode="r:gz") as tar:
        for info in tar:
            if info.name.startswith("/") or ".." in Path(info.name).parts:
                raise ValueError(f"Unsafe member path in {artifact}: {info.name!r}")
            if not (info.isdir() or info.isreg()):
                raise ValueError(
                    f"Unsupported member type in {artifact}: {info.name!r}"
                )
            root = info.name.split("/", 1)[0]
            if root not in roots:
                roots.append(root)
        if len(roots) != 1:
            raise ValueError(
                f"{artifact} must contain exactly one top-level directory, "
                f"found: {sorted(roots)}"
            )
        tar.extractall(output_dir, filter="data")

    return output_dir / roots[0]
