"""
Loading and verification of the pinned reference corpus manifest.

The manifest (``evals/corpora.json``, schema ``versta.eval-corpora/1``) is
produced by the corpus pinning task and records, per shipped direction, the
FLORES-200 dev-split source and target payload files with their SHA-256
hashes and sentence counts. This module is the consumer side: it validates
the manifest shape, resolves the payload files on disk and fails closed on
any hash or count drift, so a corrupted or swapped corpus can never
produce scores that look comparable to historical ones.
"""

import json
import re

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, NoReturn, cast

from .typing import (
    CORPORA_SCHEMA,
    CorpusDirection,
    CorpusFile,
    CorpusManifest,
)


class CorpusError(ValueError):
    """The corpus manifest or its payload files fail verification."""


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


def _fail(message: str) -> NoReturn:
    raise CorpusError(message)


def _require_string(data: Dict[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(f"{path}.{key}: expected a non-blank string")
    return value


def _require_file(data: Dict[str, Any], key: str, path: str) -> CorpusFile:
    payload = data.get(key)
    if not isinstance(payload, dict):
        _fail(f"{path}.{key}: expected an object")
    payload = cast(Dict[str, Any], payload)

    payload_path = payload.get("path")
    if not isinstance(payload_path, str) or not payload_path:
        _fail(f"{path}.{key}.path: expected a non-blank string")
    if Path(payload_path).is_absolute():
        _fail(f"{path}.{key}.path: must be relative to the corpus directory")

    digest = payload.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        _fail(f"{path}.{key}.sha256: expected a lowercase SHA-256 hex digest")

    return CorpusFile(path=payload_path, sha256=digest)


def load_manifest(manifest_path: Path) -> CorpusManifest:
    """
    Loads and validates the corpus manifest.

    Args:
        manifest_path (Path): Path to the ``corpora.json`` manifest.

    Returns:
        CorpusManifest: The validated manifest.

    Raises:
        CorpusError: If the file is missing, unparseable, or violates the
            ``versta.eval-corpora/1`` shape.
    """
    if not manifest_path.is_file():
        _fail(f"corpus manifest not found: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"corpus manifest {manifest_path} is not valid JSON: {exc}")

    if not isinstance(data, dict):
        _fail(f"corpus manifest {manifest_path}: top level must be an object")
    data = cast(Dict[str, Any], data)

    if data.get("schema") != CORPORA_SCHEMA:
        _fail(
            f"corpus manifest {manifest_path}: expected schema "
            f"'{CORPORA_SCHEMA}', got {data.get('schema')!r}"
        )

    corpus = _require_string(data, "corpus", "manifest")
    revision = _require_string(data, "revision", "manifest")
    split = _require_string(data, "split", "manifest")
    license_ = _require_string(data, "license", "manifest")

    directions = data.get("directions")
    if not isinstance(directions, dict) or not directions:
        _fail("manifest.directions: expected a non-empty object")
    directions = cast(Dict[str, Any], directions)

    typed_directions: Dict[str, CorpusDirection] = {}
    for direction, entry in directions.items():
        path = f"manifest.directions['{direction}']"
        if not isinstance(entry, dict):
            _fail(f"{path}: expected an object")
        entry = cast(Dict[str, Any], entry)

        sentences = entry.get("sentences")
        if (
            not isinstance(sentences, int)
            or isinstance(sentences, bool)
            or sentences < 1
        ):
            _fail(f"{path}.sentences: expected a positive integer")

        typed_directions[direction] = CorpusDirection(
            flores_source=_require_string(entry, "flores_source", path),
            flores_target=_require_string(entry, "flores_target", path),
            sentences=sentences,
            source=_require_file(entry, "source", path),
            target=_require_file(entry, "target", path),
        )

    return CorpusManifest(
        schema=CORPORA_SCHEMA,
        corpus=corpus,
        revision=revision,
        split=split,
        license=license_,
        directions=typed_directions,
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
            f"(see evals/README.md)"
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
        manifest (CorpusManifest): A manifest from `load_manifest`.
        direction (str): Direction key as used across the pipeline,
            ``<source>-<target>`` (e.g. ``en-es``).
        corpus_dir (Path): Root the manifest's relative payload paths resolve
            against.

    Returns:
        CorpusSlice: The verified source and reference sentences plus the
        provenance recorded into every eval result.

    Raises:
        CorpusError: If the direction is absent or a payload fails
            verification.
    """
    entry = manifest["directions"].get(direction)
    if entry is None:
        available = ", ".join(sorted(manifest["directions"]))
        _fail(
            f"corpus manifest has no entry for direction '{direction}' "
            f"(available: {available})"
        )

    flores_source = entry["flores_source"]
    flores_target = entry["flores_target"]

    source_lines = _read_verified_lines(
        corpus_dir / entry["source"]["path"],
        entry["source"]["sha256"],
        f"source corpus for {direction} ({flores_source})",
    )
    target_lines = _read_verified_lines(
        corpus_dir / entry["target"]["path"],
        entry["target"]["sha256"],
        f"target corpus for {direction} ({flores_target})",
    )

    expected = entry["sentences"]
    for side, lines in (("source", source_lines), ("target", target_lines)):
        if len(lines) != expected:
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
        corpus=manifest["corpus"],
        split=manifest["split"],
        revision=manifest["revision"],
        flores_source=flores_source,
        flores_target=flores_target,
        source_sentences=source_lines,
        target_sentences=target_lines,
        source_sha256=entry["source"]["sha256"],
        target_sha256=entry["target"]["sha256"],
    )
