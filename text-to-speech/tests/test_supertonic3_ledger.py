"""Conversion ledger: schema-stamped, append-only, atomic (NEU-49..54 conventions)."""

import json

import pytest

from versta.supertonic3.ledger import LedgerError, append_entry, load_ledger
from versta.supertonic3.typing import (
    CONVERSION_LEDGER_SCHEMA,
    LedgerEntry,
    ToolchainRecord,
    UpstreamRecord,
)


def _entry(bundle_sha: str) -> LedgerEntry:
    toolchain = ToolchainRecord(mnn_commit="d407447e", mnn_tag="3.6.1", python="3.12")
    upstream = UpstreamRecord(repo="Supertone/supertonic-3", revision="deadbeef")
    verification = {
        "schema": "versta.tts-verification/1",
        "seed": 42,
        "upstream": upstream,
        "toolchain": toolchain,
        "metrics": [],
        "thresholds_ref": "test",
        "passed": True,
    }
    return LedgerEntry(
        recorded_at="2026-10-15T00:00:00+00:00",
        version="v2.0.0",
        upstream=upstream,
        toolchain=toolchain,
        artifacts=[],
        verification=verification,
        bundle_sha256=bundle_sha,
        bundle_bytes=105000000,
        override_reason=None,
    )


def test_ledger_seeds_and_appends_atomically(tmp_path):
    path = tmp_path / "evals" / "ledger" / "supertonic3.json"
    ledger = load_ledger(path)
    assert ledger["schema"] == CONVERSION_LEDGER_SCHEMA and ledger["entries"] == []

    append_entry(path, _entry("aa"))
    append_entry(path, _entry("bb"))
    stored = load_ledger(path)
    assert [e["bundle_sha256"] for e in stored["entries"]] == ["aa", "bb"]
    assert not (tmp_path / "evals" / "ledger" / "supertonic3.json.tmp").exists()


def test_ledger_refuses_duplicate_bundle(tmp_path):
    path = tmp_path / "supertonic3.json"
    append_entry(path, _entry("aa"))
    with pytest.raises(LedgerError, match="duplicate"):
        append_entry(path, _entry("aa"))
    assert len(load_ledger(path)["entries"]) == 1


def test_ledger_rejects_foreign_schema(tmp_path):
    path = tmp_path / "supertonic3.json"
    path.write_text(json.dumps({"schema": "something/else", "entries": []}))
    with pytest.raises(LedgerError):
        load_ledger(path)
