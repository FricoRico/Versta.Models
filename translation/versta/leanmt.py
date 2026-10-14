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

A second handshake covers the engine CLI the quality harness drives: the
``versta.engine-cli/1`` batch file→file translate contract of
``leanmt-bench translate`` (NEU-50), pinned here as ``LEANMT_CLI_SPEC`` and
documented in PIPELINE.md "leanmt translate CLI". The normative semantics
live in the Versta.Leanmt README ("Batch translation (file → file)"); a
spec bump is coordinated exactly like a format bump.
"""

import struct

from pathlib import Path
from typing import IO, List, Sequence, Tuple

# Mirrors kBinaryFileVersion in Versta.Leanmt src/Io.hh (the bergamot int8
# layout). Change only via a negotiated format bump with the leanmt maintainer.
LEANMT_BINARY_VERSION = 1

# Mirrors kCliSpec in Versta.Leanmt bench/bench_translate.cc: the versioned
# contract of `leanmt-bench translate` (batch file→file mode, NEU-50). The
# eval harness refuses any engine binary whose --version does not report
# this identifier; bump only via a negotiated spec change with the leanmt
# maintainer.
LEANMT_CLI_SPEC = "versta.engine-cli/1"

# The version is the first uint64 of the binary model file, as written
# little-endian by marian/bergamot on every supported platform.
_HEADER_SIZE = 8

# Tensor-table record header, mirroring `struct Header` in Versta.Leanmt
# src/Io.hh: name_length (bytes, including the NUL terminator), tensor type,
# shape_length (number of int32 dims) and data_length (bytes) — four uint64.
_TENSOR_HEADER_SIZE = 32

# Sanity bound on the tensor count declared by a model file; real bergamot
# models carry hundreds of tensors, so anything past this is corruption.
_MAX_TENSORS = 65536

# Embedding tensors whose exact presence marks a model trained with separate
# source and target vocabularies: such models carry split
# encoder_Wemb/decoder_Wemb embeddings instead of the single shared ``Wemb``
# (see src/Io.cc in Versta.Leanmt). The auxiliary ``*_QuantMultA`` scalars do
# not count on their own — the base tensor is always co-written. NEU-26 makes
# leanmt fail fast when such a model is loaded without a target vocabulary.
_SPLIT_WEMB_TENSORS = frozenset({"encoder_Wemb", "decoder_Wemb"})


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


def _read_exact(
    stream: IO[bytes], size: int, what: str, source: str, offset: int
) -> bytes:
    """
    Reads exactly ``size`` bytes from a stream, mirroring the bounds-checked
    Cursor in Versta.Leanmt src/Io.cc: a short read is a precise corrupt-or-
    truncated error, never a partial parse.

    Args:
        stream (IO[bytes]): Readable binary stream positioned at the field.
        size (int): Bytes expected for the field.
        what (str): Field description for the error message.
        source (str): Human-readable origin of the bytes, for error messages.
        offset (int): Byte offset where the read starts, for error messages.

    Returns:
        bytes: Exactly ``size`` bytes.

    Raises:
        LeanmtFormatError: If the stream ends before ``size`` bytes.
    """
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise LeanmtFormatError(
                f"corrupt or truncated model file ({source}): reading {what} "
                f"({size} bytes) at offset {offset} exceeds the file's "
                f"remaining {len(chunks)} bytes"
            )
        chunks += chunk
    return bytes(chunks)


def read_tensor_table(stream: IO[bytes], source: str) -> Tuple[int, List[str]]:
    """
    Walks the tensor table at the start of a bergamot/leanmt binary model,
    mirroring ``load_items`` in Versta.Leanmt src/Io.cc: version, tensor
    header count, one 32-byte header per tensor and the NUL-terminated tensor
    names. Only the table is read — shapes and tensor payloads follow the
    names and are left to the native loader.

    Fails closed before interpreting anything when the embedded binary version
    is not ``LEANMT_BINARY_VERSION`` (a reader written for this layout must
    not guess at foreign layouts).

    Args:
        stream (IO[bytes]): Readable stream positioned at byte 0 of the model
            (a file handle or an open tarfile member).
        source (str): Human-readable origin of the bytes, for error messages.

    Returns:
        tuple[int, list[str]]: The embedded binary version and the tensor
        names in table order.

    Raises:
        LeanmtFormatError: On truncation, on a version other than
            ``LEANMT_BINARY_VERSION``, on an implausible header count, or on a
            malformed (empty or non-NUL-terminated) tensor name.
    """
    offset = 0

    version_bytes = stream.read(_HEADER_SIZE)
    offset += len(version_bytes)
    version = parse_binary_model_version(version_bytes, source)
    if version != LEANMT_BINARY_VERSION:
        raise LeanmtFormatError(
            f"binary model version mismatch in {source}: embedded v{version}, "
            f"leanmt expects v{LEANMT_BINARY_VERSION} (kBinaryFileVersion); "
            f"refusing to parse a model layout written for a different format"
        )

    raw = _read_exact(stream, 8, "tensor header count", source, offset)
    offset += 8
    (num_headers,) = struct.unpack("<Q", raw)
    if num_headers > _MAX_TENSORS:
        raise LeanmtFormatError(
            f"corrupt model file ({source}): tensor header count "
            f"{num_headers} exceeds the sanity bound of {_MAX_TENSORS}"
        )

    headers_raw = _read_exact(
        stream, num_headers * _TENSOR_HEADER_SIZE, "tensor headers", source, offset
    )
    offset += len(headers_raw)
    name_lengths = [
        struct.unpack_from("<Q", headers_raw, index * _TENSOR_HEADER_SIZE)[0]
        for index in range(num_headers)
    ]

    names = []
    for index, name_length in enumerate(name_lengths):
        raw_name = _read_exact(
            stream,
            name_length,
            f"tensor name at header index {index}",
            source,
            offset,
        )
        offset += name_length
        if name_length < 1 or raw_name[-1:] != b"\x00":
            raise LeanmtFormatError(
                f"corrupt model file ({source}): tensor name at header index "
                f"{index} is empty or not NUL-terminated"
            )
        names.append(raw_name[:-1].decode("utf-8", errors="replace"))

    return version, names


def read_tensor_names(path: Path) -> List[str]:
    """
    Reads the tensor names from the header of a model file on disk.

    Args:
        path (Path): Path to the binary model file.

    Returns:
        list[str]: The tensor names in table order.

    Raises:
        FileNotFoundError: If the model file does not exist.
        LeanmtFormatError: On any format violation (see `read_tensor_table`).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as handle:
        return read_tensor_table(handle, str(path))[1]


def model_requires_target_vocabulary(names: Sequence[str]) -> bool:
    """
    Whether a model carrying these tensor names needs a separate target
    vocabulary, per the leanmt loader's split-embedding rule (NEU-26): models
    trained with distinct source/target vocabularies embed ``encoder_Wemb``
    and ``decoder_Wemb`` tensors; shared-vocabulary models carry a single
    ``Wemb``.

    Args:
        names (Sequence[str]): Tensor names from `read_tensor_table`.

    Returns:
        bool: True when the model carries split-Wemb embeddings and therefore
        fails fast in the leanmt loader without a target vocabulary.
    """
    return not _SPLIT_WEMB_TENSORS.isdisjoint(names)
