import json

from pathlib import Path

import pytest

from versta.eval.gate import GateReport, evaluate_gate, gate_info
from versta.eval.ledger import (
    LEDGER_SCHEMA,
    LedgerError,
    append_entry,
    empty_ledger,
    ledger_path,
    load_ledger,
    predecessor_entry,
    validate_ledger,
    write_ledger,
)

from conftest import make_eval_result


def _recorded_ledger(*versions: str) -> dict:
    """A ledger with each version recorded as a clean gate decision."""
    ledger = empty_ledger("en-es", "en", "es")
    for index, version in enumerate(versions):
        result = make_eval_result(version=version, chrf=60.0 + index)
        report = evaluate_gate(
            result=result,
            ledger=ledger,
            ledger_path=Path("evals/ledger/en-es.json"),
            threshold=1.0,
        )
        append_entry(ledger, result, gate_info(report))
    return ledger


def test_ledger_round_trip(tmp_path: Path) -> None:
    ledger = _recorded_ledger("v1.12.0", "v1.12.14")
    path = ledger_path(tmp_path / "ledger", "en-es")
    write_ledger(ledger, path)

    loaded = load_ledger(path)
    assert loaded == ledger
    assert loaded["schema"] == LEDGER_SCHEMA
    assert loaded["direction"] == "en-es"
    assert [entry["version"] for entry in loaded["entries"]] == [
        "v1.12.0",
        "v1.12.14",
    ]
    # The full audit trail rides in each entry.
    entry = loaded["entries"][1]
    assert entry["result"]["scores"]["chrf"] == 61.0
    assert entry["result"]["corpus"]["revision"] == "tests/fixture@0001"
    assert entry["result"]["model"]["bundle_sha256"]
    assert entry["gate"]["decision"] == "pass"
    assert entry["gate"]["predecessor_version"] == "v1.12.0"
    assert entry["gate"]["delta_chrf"] == 1.0


def test_append_rejects_duplicate_version() -> None:
    ledger = _recorded_ledger("v1.12.14")
    report = evaluate_gate(
        result=make_eval_result(version="v2.0.0"),
        ledger=ledger,
        ledger_path=Path("x.json"),
        threshold=1.0,
    )
    with pytest.raises(LedgerError, match="already recorded"):
        append_entry(ledger, make_eval_result(version="v1.12.14"), gate_info(report))


def test_append_rejects_direction_mismatch() -> None:
    ledger = _recorded_ledger("v1.12.14")
    wrong = make_eval_result(direction="es-en", version="v2.0.0")
    gate = gate_info(
        GateReport(
            result=wrong,
            ledger=ledger,
            ledger_path=Path("x.json"),
            decision="pass",
            ships=True,
            threshold_chrf=1.0,
            predecessor=None,
            delta_bleu=None,
            delta_chrf=None,
            delta_comet22=None,
            reason=None,
            failure_why=None,
            decided_utc="2026-10-14T00:00:00Z",
        )
    )
    with pytest.raises(LedgerError, match="es-en.*en-es"):
        append_entry(ledger, wrong, gate)


def test_append_rejects_unversioned_result() -> None:
    ledger = empty_ledger("en-es", "en", "es")
    result = make_eval_result()
    result["model"]["version"] = "  "
    gate = gate_info(
        GateReport(
            result=result,
            ledger=ledger,
            ledger_path=Path("x.json"),
            decision="baseline",
            ships=True,
            threshold_chrf=1.0,
            predecessor=None,
            delta_bleu=None,
            delta_chrf=None,
            delta_comet22=None,
            reason=None,
            failure_why=None,
            decided_utc="2026-10-14T00:00:00Z",
        )
    )
    with pytest.raises(LedgerError, match="no.*model version"):
        append_entry(ledger, result, gate)


def test_predecessor_skips_the_candidates_own_version() -> None:
    """A re-run of an already-recorded release must compare against the
    previous version, never against itself (which would fake a zero delta)."""
    ledger = _recorded_ledger("v1.12.0", "v1.12.14")

    assert predecessor_entry(ledger, "v2.0.0")["version"] == "v1.12.14"
    assert predecessor_entry(ledger, "v1.12.14")["version"] == "v1.12.0"
    assert predecessor_entry(_recorded_ledger("v1.12.14"), "v1.12.14") is None
    assert predecessor_entry(empty_ledger("en-es", "en", "es"), "v2.0.0") is None


def test_load_ledger_missing_and_invalid(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="not found"):
        load_ledger(tmp_path / "en-es.json")

    broken = tmp_path / "en-es.json"
    broken.write_text("{ not json", encoding="utf-8")
    with pytest.raises(LedgerError, match="not valid JSON"):
        load_ledger(broken)


def test_validate_ledger_rejects_schema_violations(tmp_path: Path) -> None:
    path = tmp_path / "en-es.json"
    ledger = _recorded_ledger("v1.12.14")

    wrong_schema = dict(ledger, schema="versta.eval-ledger/0")
    with pytest.raises(LedgerError, match="versta.eval-ledger/1"):
        validate_ledger(wrong_schema, path)

    bad_decision = json.loads(json.dumps(ledger))
    bad_decision["entries"][0]["gate"]["decision"] = "yolo"
    with pytest.raises(LedgerError, match="gate.decision"):
        validate_ledger(bad_decision, path)

    # Overrides must carry a written reason — the machine-checked contract.
    no_reason = json.loads(json.dumps(ledger))
    no_reason["entries"][0]["gate"]["decision"] = "override"
    with pytest.raises(LedgerError, match="requires a written reason"):
        validate_ledger(no_reason, path)

    missing_scores = json.loads(json.dumps(ledger))
    del missing_scores["entries"][0]["result"]["scores"]["chrf"]
    with pytest.raises(LedgerError, match="scores.chrf"):
        validate_ledger(missing_scores, path)
