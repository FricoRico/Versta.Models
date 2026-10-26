"""
Loading and validation of traveler probe corpora.

A corpus is one JSON document per direction (``<src>-<tgt>.json``) holding
~40 probe sentences with human reference translations across the NEU-30 harm
classes (numbers, times, prices, addresses, directions, emergencies). The
runner fails closed on a malformed corpus — a corpus that cannot name its
direction or is missing references would make chrF meaningless.
"""

import hashlib
import json

from pathlib import Path
from typing import Any, Dict, List, NoReturn, cast

from .install import sha256_of
from .typing import Corpus


class CorpusError(ValueError):
    """A probe corpus file failed validation."""


def _fail(path: str, message: str) -> NoReturn:
    raise CorpusError(f"{path}: {message}")


def _require_keys(data: Dict[str, Any], keys: tuple, path: str) -> None:
    for key in keys:
        if key not in data:
            _fail(path, f"missing required field '{key}'")


def load_corpus(corpus_path: Path) -> Corpus:
    """
    Loads and validates one probe corpus file.

    Args:
            corpus_path (Path): ``<src>-<tgt>.json`` corpus document.

    Returns:
        Corpus: the validated corpus.

    Raises:
        CorpusError: on any malformed field; FileNotFoundError when the file
            does not exist.
    """
    with open(corpus_path, "r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if not isinstance(data, dict):
        _fail(corpus_path.name, "expected a JSON object")
    return validate_corpus(data, path=corpus_path.name)


def validate_corpus(data: Any, path: str = "corpus") -> Corpus:
    """
    Validates a parsed probe corpus document against the runner's contract:
    non-empty direction codes, at least one entry, every entry with non-blank
    ``id``/``source``/``reference``, a numeric ``chrf_floor`` in (0, 100], and
    entries tagged with at least one harm class.
    """
    if not isinstance(data, dict):
        _fail(path, "expected a JSON object")
    _require_keys(
        data,
        (
            "name",
            "source_language",
            "target_language",
            "version",
            "chrf_floor",
            "entries",
        ),
        path,
    )

    for key in ("name", "source_language", "target_language"):
        if not isinstance(data[key], str) or not data[key].strip():
            _fail(f"{path}.{key}", "expected non-blank string")

    if not isinstance(data["version"], int) or data["version"] < 1:
        _fail(f"{path}.version", "expected positive integer")

    floor = data["chrf_floor"]
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        _fail(f"{path}.chrf_floor", "expected number")
    if not 0.0 < float(floor) <= 100.0:
        _fail(f"{path}.chrf_floor", f"expected chrF floor in (0, 100], got {floor}")

    entries = data["entries"]
    if not isinstance(entries, list) or not entries:
        _fail(f"{path}.entries", "expected non-empty array")
    seen_ids = set()
    digits_seen = 0
    for index, entry in enumerate(entries):
        entry_path = f"{path}.entries[{index}]"
        if not isinstance(entry, dict):
            _fail(entry_path, "expected object")
        _require_keys(entry, ("id", "tags", "source", "reference"), entry_path)
        for key in ("id", "source", "reference"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                _fail(f"{entry_path}.{key}", "expected non-blank string")
        if not isinstance(entry["tags"], list) or not entry["tags"]:
            _fail(f"{entry_path}.tags", "expected non-empty array")
        if entry["id"] in seen_ids:
            _fail(f"{entry_path}.id", f"duplicate entry id {entry['id']!r}")
        seen_ids.add(entry["id"])
        if any(ch.isdigit() for ch in entry["source"]):
            digits_seen += 1

    # The corpus exists to measure the NEU-30 harm class; a corpus that stops
    # carrying digits silently disarms the digit-preservation gate.
    if digits_seen * 2 < len(entries):
        _fail(
            f"{path}.entries",
            f"only {digits_seen}/{len(entries)} entries carry digits; "
            "at least half must, or the digit-preservation gate cannot see "
            "the harm class it guards",
        )

    return cast(Corpus, data)


def corpus_payload_sha256(corpus_paths: List[Path]) -> str:
    """
    Pins the exact corpus payloads a run scored against: sha256 over the
    per-file ``"<name>:<file sha256>"`` lines, sorted by file name. The
    scorecard records the digest in ``environment.corpus_sha256`` so a card
    can be tied to the precise corpus bytes, not just the corpus name —
    same artifact-pinning discipline as the bundle sha256.

    Args:
        corpus_paths (List[Path]): the corpus files loaded for the run.

    Returns:
        str: lowercase hex digest; order of ``corpus_paths`` is irrelevant.

    Raises:
        ValueError: on an empty list or duplicate file names (ambiguous
            payload description).
    """
    if not corpus_paths:
        raise ValueError("corpus payload hash requires at least one corpus file")
    if len({path.name for path in corpus_paths}) != len(corpus_paths):
        raise ValueError("duplicate corpus file names make the payload hash ambiguous")
    lines = sorted(f"{path.name}:{sha256_of(path)}" for path in corpus_paths)
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def corpus_path_for(corpus_dir: Path, source: str, target: str) -> Path:
    """
    Resolves the corpus file for one direction inside a corpus directory.

    Args:
        corpus_dir (Path): directory holding ``<src>-<tgt>.json`` files.
        source (str): source language code.
        target (str): target language code.

    Returns:
        Path: the corpus file to load.
    """
    return corpus_dir / f"{source}-{target}.json"
