"""
Consumer side of the corpus contract: verified slices for the eval harness.

The producer side (``pipeline.py``) builds, fetches and verifies the pinned
snapshot as a whole; this module is what the scoring harness actually reads:
one direction's source and reference sentences, re-verified against the
checked-in manifest at the moment of use. A corrupted or swapped payload
fails closed here even if it slipped into a populated cache directory, so
an untrusted corpus can never produce scores that look comparable to
historical ones.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import List, NoReturn

from .manifest import load_manifest as _load_validated_manifest
from .typing import CorpusManifest, ManifestDirection


class CorpusError(ValueError):
    """The corpus manifest or its payload files fail verification."""


def _fail(message: str) -> NoReturn:
    raise CorpusError(message)


def load_manifest(manifest_path: Path) -> CorpusManifest:
    """
    Loads and validates the checked-in corpus manifest for harness use.

    Thin wrapper over the producer loader (``manifest.load_manifest``, which
    validates fail-fast) that normalizes every failure — missing file,
    unparseable JSON, schema violation — into ``CorpusError``.

    Args:
        manifest_path (Path): Path to the manifest (e.g.
            ``evals/corpora/flores200-dev.json``).

    Returns:
        CorpusManifest: The validated manifest.

    Raises:
        CorpusError: If the file is missing, unparseable, or violates the
            ``versta/corpus-manifest@1`` shape.
    """
    if not manifest_path.is_file():
        _fail(f"corpus manifest not found: {manifest_path}")
    try:
        return _load_validated_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise CorpusError(str(exc)) from exc


@dataclass(frozen=True)
class CorpusSlice:
    """One direction's verified corpus payload, ready for translation."""

    direction: str
    corpus: str
    split: str
    revision: str
    flores_source: str
    flores_target: str
    source_sentences: List[str]
    target_sentences: List[str]
    source_sha256: str
    target_sha256: str


def snapshot_dir(manifest: CorpusManifest, corpus_root: Path) -> Path:
    """
    Resolves the directory a corpus fetch materializes for this manifest.

    Args:
        manifest (CorpusManifest): The validated checked-in manifest.
        corpus_root (Path): The fetch cache root (``output/corpora`` by
            default; payload paths never come from the manifest itself, so a
            tampered manifest cannot redirect reads outside the cache).

    Returns:
        Path: ``<corpus_root>/<name>-r<revision>/``.
    """
    return Path(corpus_root) / f"{manifest['name']}-r{manifest['revision']}"


def _find_direction(manifest: CorpusManifest, direction: str) -> ManifestDirection:
    for entry in manifest["directions"]:
        if f"{entry['source']}-{entry['target']}" == direction:
            return entry
    available = ", ".join(
        sorted(
            f"{entry['source']}-{entry['target']}" for entry in manifest["directions"]
        )
    )
    _fail(
        f"corpus manifest has no entry for direction '{direction}' "
        f"(available: {available})"
    )


def _read_verified_lines(file_path: Path, expected_sha256: str, what: str) -> List[str]:
    """
    Reads a corpus payload file, verifying its pinned SHA-256 before use.

    Lines are returned without their terminators; interior whitespace is
    untouched (metrics must see the corpus byte-for-byte as pinned).

    Args:
        file_path (Path): Payload file on disk.
        expected_sha256 (str): Hash pinned in the manifest.
        what (str): Human-readable description, for error messages.

    Returns:
        list[str]: The corpus lines.

    Raises:
        CorpusError: If the file is missing or its hash differs from the pin.
    """
    if not file_path.is_file():
        _fail(
            f"{what} not found: {file_path} — fetch the pinned corpus first "
            f"(uv run python -m versta.eval.corpora fetch)"
        )
    blob = file_path.read_bytes()
    actual = sha256(blob).hexdigest()
    if actual != expected_sha256:
        _fail(
            f"hash mismatch for {what} ({file_path}): pinned "
            f"{expected_sha256}, file has {actual} — refusing to score "
            f"against an unpinned corpus"
        )
    return blob.decode("utf-8").splitlines()


def resolve_slice(
    manifest: CorpusManifest, direction: str, corpus_dir: Path
) -> CorpusSlice:
    """
    Resolves, verifies and reads the corpus payload for one direction.

    Args:
        manifest (CorpusManifest): A manifest from
            ``versta.eval.corpora.manifest.load_manifest`` (fail-fast
            validated, so paths and cross-references are already sound).
        direction (str): Direction key as used across the pipeline,
            ``<source>-<target>`` (e.g. ``en-es``).
        corpus_dir (Path): Root of the extracted snapshot (the directory
            returned by ``versta.eval.corpora fetch``).

    Returns:
        CorpusSlice: The verified source and reference sentences plus the
        provenance recorded into every eval result.

    Raises:
        CorpusError: If the direction is absent or a payload fails
            verification.
    """
    entry = _find_direction(manifest, direction)
    source_file = entry["source_file"]
    target_file = entry["target_file"]
    source_pin = manifest["files"][source_file]
    target_pin = manifest["files"][target_file]

    flores_source = manifest["language_map"][entry["source"]]
    flores_target = manifest["language_map"][entry["target"]]

    source_lines = _read_verified_lines(
        corpus_dir / source_file,
        source_pin["sha256"],
        f"source corpus for {direction} ({flores_source})",
    )
    target_lines = _read_verified_lines(
        corpus_dir / target_file,
        target_pin["sha256"],
        f"target corpus for {direction} ({flores_target})",
    )

    expected = entry["sentences"]
    for side, lines, pin in (
        ("source", source_lines, source_pin),
        ("target", target_lines, target_pin),
    ):
        if len(lines) != expected or len(lines) != pin["sentences"]:
            _fail(
                f"{side} corpus for {direction}: manifest records {expected} "
                f"sentences, file carries {len(lines)} — the manifest entry "
                f"and payload disagree"
            )
    if any(not line.strip() for line in source_lines):
        _fail(
            f"source corpus for {direction} contains blank lines; the harness "
            f"requires exactly one sentence per line to keep engine output "
            f"line-aligned"
        )

    return CorpusSlice(
        direction=direction,
        corpus=manifest["name"],
        split=manifest["split"],
        revision=f"r{manifest['revision']}",
        flores_source=flores_source,
        flores_target=flores_target,
        source_sentences=source_lines,
        target_sentences=target_lines,
        source_sha256=source_pin["sha256"],
        target_sha256=target_pin["sha256"],
    )
