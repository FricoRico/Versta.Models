"""
Handshake with the leanmt on-device inference loader (Versta.Leanmt).

leanmt mmaps the Bergamot binary model file and reads a uint64 binary format
version as the very first field (``kBinaryFileVersion`` in ``src/Io.hh`` of
Versta.Leanmt) and fails fast when it differs from what the loader was built
for. That check only fires once the app has crossed JNI, so every bundle
manifest this pipeline produces also declares ``leanmt_binary_version`` in the
bundle-level ``metadata.json``: the app can pre-flight a bundle before
touching native code and surface a clean error on stale bundles.

This module is the converter side of that contract and fails closed: download
and bundle abort unless every model file's embedded header version equals
``LEANMT_BINARY_VERSION`` exactly, and the schema gate refuses manifests that
declare anything else. Contract counterpart: "Model loading contract" in the
Versta.Leanmt README. Key naming and semantics are negotiated with Elin
Sørensen (NEU-25); app-side pre-flight is owned by Maya Lindqvist.
"""

import struct

from pathlib import Path

# Mirrors kBinaryFileVersion in Versta.Leanmt src/Io.hh (the bergamot int8
# layout). Change only via a negotiated format bump with the leanmt maintainer.
LEANMT_BINARY_VERSION = 1

# The version is the first uint64 of the binary model file, as written
# little-endian by marian/bergamot on every supported platform.
_HEADER_SIZE = 8


class LeanmtFormatError(ValueError):
    """A model file violates the leanmt binary format contract."""


def parse_binary_model_version(header: bytes, source: str) -> int:
    """
    Parses the embedded binary format version from the leading bytes of a
    bergamot/leanmt binary model file.

    Args:
        header (bytes): At least the first 8 bytes of the model file.
        source (str): Human-readable origin of the bytes, for error messages.

    Returns:
        int: The embedded binary format version.

    Raises:
        LeanmtFormatError: If fewer than 8 bytes are available (truncated file).
    """
    if len(header) < _HEADER_SIZE:
        raise LeanmtFormatError(
            f"truncated model file ({source}): only {len(header)} byte(s) "
            f"readable, need {_HEADER_SIZE} for the binary file version header"
        )
    return int(struct.unpack("<Q", header[:_HEADER_SIZE])[0])


def read_binary_model_version(path: Path) -> int:
    """
    Reads the embedded binary format version of a model file on disk.

    Args:
        path (Path): Path to the binary model file.

    Returns:
        int: The embedded binary format version.

    Raises:
        FileNotFoundError: If the model file does not exist.
        LeanmtFormatError: If the file is too short to carry a version header.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as handle:
        return parse_binary_model_version(handle.read(_HEADER_SIZE), str(path))


def assert_leanmt_compatible(path: Path) -> None:
    """
    Fails closed unless the model file's embedded binary version matches the
    leanmt loader contract exactly (``LEANMT_BINARY_VERSION``).

    Args:
        path (Path): Path to the binary model file.

    Raises:
        FileNotFoundError: If the model file does not exist.
        LeanmtFormatError: If the header is truncated or the embedded version
            is not ``LEANMT_BINARY_VERSION``.
    """
    version = read_binary_model_version(path)
    if version != LEANMT_BINARY_VERSION:
        raise LeanmtFormatError(
            f"binary model version mismatch in {path}: embedded v{version}, "
            f"leanmt expects v{LEANMT_BINARY_VERSION} (kBinaryFileVersion); "
            f"refusing to ship this model"
        )
