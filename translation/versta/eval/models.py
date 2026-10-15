"""
Resolving the artifact under evaluation.

The harness scores the artifacts the pipeline actually ships, in either of
two forms:

  * a direction directory as produced by ``versta.download``
    (``--direction_dir``) — the pre-bundling form, handy while iterating;
  * a shipped bundle tarball plus a direction (``--bundle``) — the exact
    artifact the app side-loads, extracted read-only into the run's work
    directory.

Both paths re-validate the metadata against the app's loader schema
(:mod:`versta.schema`) before any engine runs: an artifact that would fail
on-device validation is an eval failure, not a score.
"""

import json
import tarfile

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, cast

from ..schema import validate_bundle_metadata, validate_direction_metadata


class ModelInputError(ValueError):
    """The artifact under evaluation is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class DirectionInput:
    """A verified on-disk direction, engine-ready."""

    direction: str
    source_language: str
    target_language: str
    architecture: str
    base_model: str
    version: str
    model_path: Path
    vocabulary_path: Path
    target_vocabulary_path: Optional[Path]
    shortlist_path: Path
    metadata_sha256: str
    model_sha256: str
    bundle_file: Optional[str]
    bundle_sha256: Optional[str]
    # Bytes of the artifact under evaluation: tarball size for bundle runs,
    # sum of the model/vocabulary/shortlist payloads for direction runs.
    size_bytes: int


def sha256_file(path: Path) -> str:
    """
    Computes the SHA-256 hex digest of a file.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hex digest.
    """
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_direction_metadata(direction_dir: Path) -> dict:
    metadata_file = direction_dir / "metadata.json"
    if not metadata_file.is_file():
        raise ModelInputError(f"no metadata.json in {direction_dir}")
    try:
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelInputError(
            f"metadata.json in {direction_dir} is not valid JSON: {exc}"
        )
    return cast(dict, data)


def _direction_input(
    *,
    direction_dir: Path,
    bundle_file: Optional[str],
    bundle_sha256: Optional[str],
    size_bytes: Optional[int],
) -> DirectionInput:
    """
    Builds a DirectionInput from an on-disk direction directory, validating
    the metadata and file payloads against the app loader schema first.
    """
    metadata = _load_direction_metadata(direction_dir)

    # Validate against the exact contract the app's loader expects; also
    # verifies every referenced file exists on disk.
    validate_direction_metadata(metadata, base_dir=direction_dir)

    source_language = str(metadata["source_language"])
    target_language = str(metadata["target_language"])
    direction = f"{source_language}-{target_language}"
    files = metadata["files"]

    def payload(key: str) -> Path:
        return direction_dir / str(files[key])

    vocabulary_path = payload("vocabulary")
    target_vocabulary_path = (
        payload("target_vocabulary") if files.get("target_vocabulary") else None
    )
    model_path = payload("model")

    payload_paths = [model_path, vocabulary_path, payload("shortlist")]
    if target_vocabulary_path is not None:
        payload_paths.append(target_vocabulary_path)

    return DirectionInput(
        direction=direction,
        source_language=source_language,
        target_language=target_language,
        architecture=str(metadata["architecture"]),
        base_model=str(metadata["base_model"]),
        version=str(metadata["version"]),
        model_path=model_path,
        vocabulary_path=vocabulary_path,
        target_vocabulary_path=target_vocabulary_path,
        shortlist_path=payload("shortlist"),
        metadata_sha256=sha256_file(direction_dir / "metadata.json"),
        model_sha256=sha256_file(model_path),
        bundle_file=bundle_file,
        bundle_sha256=bundle_sha256,
        # For bundle runs the caller passes the tarball size (the artifact
        # the app downloads); for direction runs the payloads are the size.
        size_bytes=(
            size_bytes
            if size_bytes is not None
            else sum(path.stat().st_size for path in payload_paths)
        ),
    )


def load_direction_dir(direction_dir: Path) -> DirectionInput:
    """
    Resolves a ``versta.download`` output directory as the eval target.

    Args:
        direction_dir (Path): Directory containing metadata.json and the
            model payloads.

    Returns:
        DirectionInput: Verified, engine-ready view of the direction.

    Raises:
        ModelInputError: If the directory is missing or has no metadata.
        ValidationError: If the metadata violates the app loader schema.
    """
    if not direction_dir.is_dir():
        raise ModelInputError(f"direction directory not found: {direction_dir}")
    return _direction_input(
        direction_dir=direction_dir,
        bundle_file=None,
        bundle_sha256=None,
        size_bytes=None,
    )


def load_from_bundle(
    bundle_path: Path, direction: str, extract_dir: Path
) -> DirectionInput:
    """
    Extracts a shipped bundle tarball and resolves one direction from it.

    The bundle-level manifest is validated against the app's bundle loader
    schema, and only the requested direction's members are unpacked
    (path-sanitized), so a hostile archive cannot write outside
    ``extract_dir``.

    Args:
        bundle_path (Path): The ``<langs>-bundle.tar.gz`` the app ships.
        direction (str): Direction to evaluate, ``<source>-<target>``.
        extract_dir (Path): Scratch directory to unpack into.

    Returns:
        DirectionInput: Verified, engine-ready view of the direction.

    Raises:
        ModelInputError: If the archive lacks the bundle manifest or the
            requested direction.
        ValidationError: If bundle or direction metadata violates the app
            loader schema.
    """
    if not bundle_path.is_file():
        raise ModelInputError(f"bundle not found: {bundle_path}")

    bundle_sha256 = sha256_file(bundle_path)
    extract_dir.mkdir(parents=True, exist_ok=True)

    direction_prefix = f"{direction}/"
    have_manifest = False
    have_direction = False

    with tarfile.open(bundle_path, "r:gz") as archive:
        for member in archive:
            if member.name == "metadata.json":
                have_manifest = True
            elif member.name.startswith(direction_prefix):
                if member.name == f"{direction}/metadata.json":
                    have_direction = True
            else:
                continue

            # Members of our own bundles always carry plain relative paths;
            # refuse anything that could escape the extraction root anyway.
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ModelInputError(
                    f"unsafe member path in {bundle_path}: {member.name!r}"
                )
            archive.extract(member, extract_dir, filter="data")

    if not have_manifest:
        raise ModelInputError(
            f"bundle {bundle_path} carries no bundle-level metadata.json"
        )
    if not have_direction:
        raise ModelInputError(
            f"bundle {bundle_path} carries no direction '{direction}'"
        )

    bundle_metadata = json.loads(
        (extract_dir / "metadata.json").read_text(encoding="utf-8")
    )
    validate_bundle_metadata(bundle_metadata)

    return _direction_input(
        direction_dir=extract_dir / direction,
        bundle_file=bundle_path.name,
        bundle_sha256=bundle_sha256,
        size_bytes=bundle_path.stat().st_size,
    )
