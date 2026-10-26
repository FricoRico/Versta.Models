"""
Bundle install: the pack-acceptance runner consumes a bundle artifact exactly
the way the app does — checksum verification, then the metadata/parse path
whose shapes and fail-closed rules are mirrored by ``versta.schema`` from the
app's Kotlin loader (``LanguageBundleMetadata`` / ``LanguageModelMetadata`` /
``files.isValid(root)``), then extraction into a staging directory.

The archive gate (``validate_bundle_archive``) runs on the untouched tarball
first: if the artifact fails the app's contract, probing stops here — a pack
that cannot be parsed is a publish defect, not a probe measurement.
"""

import hashlib
import json
import tarfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..schema import (
    validate_bundle_archive,
    validate_bundle_metadata,
    validate_direction_metadata,
)


class InstallError(ValueError):
    """A bundle artifact failed verification or install."""


@dataclass(frozen=True)
class InstalledDirection:
    """One installed, contract-validated bundle direction on disk."""

    directory: Path
    source_language: str
    target_language: str
    model: Path
    vocabulary: Path
    target_vocabulary: Path
    shortlist: Path
    encoder_layers: int
    decoder_layers: int
    ffn_depth: int
    num_heads: int


def sha256_of(path: Path) -> str:
    """
    Computes the SHA-256 of a file, streaming.

    Args:
        path (Path): file to hash.

    Returns:
        str: lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_sidecar(archive: Path) -> str:
    """
    Verifies a bundle archive against its ``.sha256`` sidecar (the checksum
    file format ``versta.bundle`` writes: ``<hex>  <name>``), matching what
    the app verifies at download time before it ever installs.

    Args:
        archive (Path): the bundle tarball; expects ``<archive minus .gz>
            .sha256`` (``en-es-bundle.tar.sha256``) next to it.

    Returns:
        str: the verified hex digest.

    Raises:
        InstallError: sidecar missing or digest mismatch.
    """
    sidecar = Path(str(archive.with_suffix("")) + ".sha256")
    if not sidecar.is_file():
        raise InstallError(f"checksum sidecar missing next to {archive.name}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    actual = sha256_of(archive)
    if actual != expected:
        raise InstallError(
            f"sha256 mismatch for {archive.name}: sidecar says {expected}, actual {actual}"
        )
    return actual


def install_bundle(archive: Path, staging_dir: Path) -> "InstalledBundle":
    """
    Installs one bundle tarball for probing: checksum verification (when the
    sidecar exists next to the archive — CI always stages it), the full
    archive contract gate, extraction under ``staging_dir``, and per-direction
    metadata validation with on-disk file existence (the app's
    ``files.isValid(root)`` rule).

    Args:
        archive (Path): the bundle tarball.
        staging_dir (Path): directory to extract into (created if missing).

    Returns:
        InstalledBundle: bundle manifest fields plus the installed directions.

    Raises:
        InstallError: on checksum/extraction failures; ValidationError
            (versta.schema) on any contract violation.
    """
    sidecar = Path(str(archive.with_suffix("")) + ".sha256")
    digest = (
        verify_checksum_sidecar(archive) if sidecar.is_file() else sha256_of(archive)
    )

    # The mirror of the app's parse path — fail-closed, first violation wins.
    validate_bundle_archive(archive)

    if staging_dir.exists():
        for child in sorted(staging_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Bundles are produced by this pipeline and already require safe relative
    # member names; the "data" filter additionally rejects device/symlink and
    # traversal payloads in hand-made (adversarial) archives.
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging_dir, filter="data")

    with open(staging_dir / "metadata.json", "r", encoding="utf-8") as handle:
        bundle_metadata: Dict[str, Any] = json.load(handle)
    validate_bundle_metadata(bundle_metadata, path=f"{archive.name}:metadata.json")

    directions: List[InstalledDirection] = []
    for entry in bundle_metadata["metadata"]:
        directory = staging_dir / entry["directory"]
        direction_meta_path = directory / "metadata.json"
        try:
            with open(direction_meta_path, "r", encoding="utf-8") as handle:
                direction_metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise InstallError(
                f"{entry['directory']}/metadata.json unreadable: {error}"
            ) from error
        validate_direction_metadata(
            direction_metadata, base_dir=directory, path=str(direction_meta_path)
        )

        files = direction_metadata["files"]
        config = direction_metadata["config"]
        directions.append(
            InstalledDirection(
                directory=directory,
                source_language=direction_metadata["source_language"],
                target_language=direction_metadata["target_language"],
                model=directory / files["model"],
                vocabulary=directory / files["vocabulary"],
                # Shared-vocabulary models declare no target vocabulary; the
                # bridge maps that to the empty string, like the app's
                # LeanmtPackage does.
                target_vocabulary=(
                    directory / files["target_vocabulary"]
                    if files.get("target_vocabulary")
                    else Path("")
                ),
                shortlist=directory / files["shortlist"],
                encoder_layers=int(config["encoder_layers"]),
                decoder_layers=int(config["decoder_layers"]),
                ffn_depth=int(config.get("ffn_depth", 2)),
                num_heads=int(config["num_heads"]),
            )
        )

    return InstalledBundle(
        archive=archive,
        sha256=digest,
        slug="-".join(bundle_metadata["languages"]),
        bundle_version=bundle_metadata["version"],
        leanmt_binary_version=bundle_metadata["leanmt_binary_version"],
        bidirectional=bundle_metadata["bidirectional"],
        directions=directions,
    )


@dataclass(frozen=True)
class InstalledBundle:
    """An installed, contract-validated bundle ready for probing."""

    archive: Path
    sha256: str
    slug: str
    bundle_version: str
    leanmt_binary_version: int
    bidirectional: bool
    directions: List[InstalledDirection]
