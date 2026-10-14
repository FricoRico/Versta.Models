"""
The checked-in ``evals/`` ledger: per-direction history of shipped versions.

One JSON file per direction lives under ``evals/ledger/<direction>.json``
(schema ``versta.eval-ledger/1``). Each entry embeds the full
``versta.eval-result/1`` document for a shipped version plus the gate
decision that admitted it, so the ledger alone answers both "what did this
version score?" and "why was it allowed to ship?" — scores, corpus
revision, engine version, bundle checksum, artifact size, and any override
reason all ride in the same record.

The ledger is **append-only and checked in**: it is the model-registry
history the pipeline charter requires, and the predecessor source the
release gate compares every refresh against. Entries are never edited or
removed; a mistaken record is corrected by a new entry, not by rewriting
history. Writes are atomic (write + rename), so a half-written ledger can
never be picked up by a gate run.
"""

import json
import os

from pathlib import Path
from typing import Optional

from .typing import (
    EVAL_RESULT_SCHEMA,
    GATE_DECISIONS,
    LEDGER_SCHEMA,
    DirectionLedger,
    EvalResult,
    GateInfo,
    LedgerEntry,
)


class LedgerError(ValueError):
    """A ledger file is missing, malformed, or an append would corrupt history."""


def ledger_path(ledger_dir: Path, direction: str) -> Path:
    """
    Returns the ledger file for one direction.

    Args:
        ledger_dir (Path): Directory holding the per-direction ledgers
            (conventionally ``evals/ledger``).
        direction (str): Direction key, ``<source>-<target>``.

    Returns:
        Path: ``<ledger_dir>/<direction>.json``.
    """
    return ledger_dir / f"{direction}.json"


def empty_ledger(
    direction: str, source_language: str, target_language: str
) -> DirectionLedger:
    """
    Builds an empty in-memory ledger for a direction that has never shipped
    a recorded version.

    Args:
        direction (str): Direction key, ``<source>-<target>``.
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        DirectionLedger: Schema-valid ledger with no entries.
    """
    return DirectionLedger(
        schema=LEDGER_SCHEMA,
        direction=direction,
        source_language=source_language,
        target_language=target_language,
        entries=[],
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerError(message)


def _require_str(data: dict, key: str, path: str) -> None:
    _require(
        isinstance(data.get(key), str) and bool(str(data.get(key)).strip()),
        f"{path}.{key} must be a non-empty string",
    )


def _validate_entry(entry: object, path: str) -> None:
    _require(isinstance(entry, dict), f"{path} must be an object")
    assert isinstance(entry, dict)  # for the type checker
    _require_str(entry, "version", path)

    result = entry.get("result")
    _require(isinstance(result, dict), f"{path}.result must be an object")
    assert isinstance(result, dict)  # for the type checker
    _require(
        result.get("schema") == EVAL_RESULT_SCHEMA,
        f"{path}.result.schema must be {EVAL_RESULT_SCHEMA!r} "
        f"(got {result.get('schema')!r})",
    )
    _require_str(result, "direction", f"{path}.result")
    model = result.get("model")
    _require(isinstance(model, dict), f"{path}.result.model must be an object")
    assert isinstance(model, dict)  # for the type checker
    _require_str(model, "version", f"{path}.result.model")
    scores = result.get("scores")
    _require(isinstance(scores, dict), f"{path}.result.scores must be an object")
    assert isinstance(scores, dict)  # for the type checker
    for metric in ("bleu", "chrf"):
        value = scores.get(metric)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{path}.result.scores.{metric} must be a number",
        )
    corpus = result.get("corpus")
    _require(isinstance(corpus, dict), f"{path}.result.corpus must be an object")
    assert isinstance(corpus, dict)  # for the type checker
    _require_str(corpus, "revision", f"{path}.result.corpus")

    gate = entry.get("gate")
    _require(isinstance(gate, dict), f"{path}.gate must be an object")
    assert isinstance(gate, dict)  # for the type checker
    _require(
        gate.get("decision") in GATE_DECISIONS,
        f"{path}.gate.decision must be one of {list(GATE_DECISIONS)} "
        f"(got {gate.get('decision')!r})",
    )
    threshold = gate.get("threshold_chrf")
    _require(
        isinstance(threshold, (int, float)) and not isinstance(threshold, bool),
        f"{path}.gate.threshold_chrf must be a number",
    )
    _require_str(gate, "decided_utc", f"{path}.gate")
    reason = gate.get("reason")
    _require(
        reason is None or isinstance(reason, str),
        f"{path}.gate.reason must be a string or null",
    )
    _require(
        gate.get("decision") not in ("override", "rebaseline")
        or bool(str(reason or "").strip()),
        f"{path}.gate: decision {gate.get('decision')!r} requires a written reason",
    )


def validate_ledger(data: object, path: Path) -> DirectionLedger:
    """
    Fail-fast structural validation of a parsed ledger document.

    Args:
        data (object): The parsed JSON document.
        path (Path): Where the document came from (for error messages).

    Returns:
        DirectionLedger: The same document, typed.

    Raises:
        LedgerError: On any schema violation. A corrupt checked-in ledger is
            a stop-the-line event: the gate must never compare against
            history it cannot trust.
    """
    _require(isinstance(data, dict), f"{path}: ledger must be a JSON object")
    assert isinstance(data, dict)  # for the type checker
    _require(
        data.get("schema") == LEDGER_SCHEMA,
        f"{path}: schema must be {LEDGER_SCHEMA!r} (got {data.get('schema')!r})",
    )
    for key in ("direction", "source_language", "target_language"):
        _require_str(data, key, str(path))
    entries = data.get("entries")
    _require(isinstance(entries, list), f"{path}: entries must be a list")
    assert isinstance(entries, list)  # for the type checker
    for index, entry in enumerate(entries):
        _validate_entry(entry, f"{path}: entries[{index}]")
    return data  # type: ignore[return-value]


def load_ledger(path: Path) -> DirectionLedger:
    """
    Loads and validates a direction ledger.

    Args:
        path (Path): The ledger file.

    Returns:
        DirectionLedger: The validated ledger.

    Raises:
        LedgerError: If the file is missing, not JSON, or schema-invalid.
    """
    if not path.is_file():
        raise LedgerError(f"ledger not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LedgerError(f"{path} is not valid JSON: {exc}")
    return validate_ledger(data, path)


def write_ledger(ledger: DirectionLedger, path: Path) -> Path:
    """
    Writes the ledger atomically (sorted keys, trailing newline) — a
    half-written ledger must never be picked up by a gate run.

    Args:
        ledger (DirectionLedger): The ledger to persist.
        path (Path): Destination file.

    Returns:
        Path: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    os.replace(temp_path, path)
    return path


def append_entry(
    ledger: DirectionLedger, result: EvalResult, gate: GateInfo
) -> DirectionLedger:
    """
    Appends one shipped version to the ledger (in place; returns it).

    Args:
        ledger (DirectionLedger): The ledger to extend.
        result (EvalResult): The eval result of the shipped candidate.
        gate (GateInfo): The gate decision that admitted it.

    Returns:
        DirectionLedger: The extended ledger.

    Raises:
        LedgerError: If the result belongs to another direction, carries no
            model version, or that version is already recorded. History is
            append-only: re-recording a version refuses loudly instead of
            silently forking the record.
    """
    version = str(result["model"].get("version", "")).strip()
    if not version:
        raise LedgerError(
            f"cannot record {result.get('direction')}: the result carries no "
            f"model version — ledger history must be attributable"
        )
    if result["direction"] != ledger["direction"]:
        raise LedgerError(
            f"result is for {result['direction']} but the ledger tracks "
            f"{ledger['direction']}"
        )
    if any(entry["version"] == version for entry in ledger["entries"]):
        raise LedgerError(
            f"{ledger['direction']} {version} is already recorded — the "
            f"ledger is append-only; record a new version instead of "
            f"rewriting history"
        )
    ledger["entries"].append(LedgerEntry(version=version, result=result, gate=gate))
    return ledger


def predecessor_entry(
    ledger: DirectionLedger, candidate_version: str
) -> Optional[LedgerEntry]:
    """
    Finds the entry a candidate must be compared against: the most recently
    recorded *other* version of the same direction.

    Skipping the candidate's own version keeps a re-run of an
    already-recorded release from comparing the model against itself.

    Args:
        ledger (DirectionLedger): The direction's ledger.
        candidate_version (str): The version under evaluation.

    Returns:
        LedgerEntry | None: The predecessor, or None when the direction has
            no recorded history beyond (possibly) the candidate itself.
    """
    for entry in reversed(ledger["entries"]):
        if entry["version"] != candidate_version:
            return entry
    return None
