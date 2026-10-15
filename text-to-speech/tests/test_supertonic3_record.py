"""The `record` step: gate-checked, self-contained ledger entries per shipped bundle."""

import json

import pytest

from versta.supertonic3.ledger import LedgerError, load_ledger
from versta.supertonic3.record_ledger import record


def _workspace(tmp_path, passed=True):
    converted = tmp_path / "converted"
    converted.mkdir(parents=True)
    upstream = {"repo": "Supertone/supertonic-3", "revision": "deadbeef"}
    toolchain = {"mnn_commit": "d407447e", "mnn_tag": "3.6.1", "python": "3.12"}
    (converted / "conversion-record.json").write_text(
        json.dumps(
            {
                "converted_at": "2026-10-15T00:00:00+00:00",
                "upstream": upstream,
                "toolchain": toolchain,
                "artifacts": [
                    {"path": "int8/vocoder.mnn", "sha256": "cc", "bytes": 42}
                ],
            }
        )
    )
    (converted / "verification.json").write_text(
        json.dumps(
            {
                "schema": "versta.tts-verification/1",
                "seed": 42,
                "upstream": upstream,
                "toolchain": toolchain,
                "metrics": [],
                "thresholds_ref": "test",
                "passed": passed,
            }
        )
    )
    tarball = tmp_path / "supertonic3-bundle.tar.gz"
    tarball.write_bytes(b"tarball-bytes")
    return converted, tarball


def test_record_appends_gate_passed_bundle(tmp_path):
    converted, tarball = _workspace(tmp_path)
    ledger = tmp_path / "evals" / "ledger" / "supertonic3.json"
    record(converted, tarball, "v2.0.0", ledger)

    (entry,) = load_ledger(ledger)["entries"]
    assert entry["version"] == "v2.0.0"
    assert entry["upstream"]["revision"] == "deadbeef"
    assert entry["verification"]["passed"] is True
    assert entry["bundle_bytes"] == len(b"tarball-bytes")
    assert entry["override_reason"] is None

    with pytest.raises(LedgerError, match="duplicate"):
        record(converted, tarball, "v2.0.0", ledger)


def test_record_refuses_failed_gate_without_override(tmp_path):
    converted, tarball = _workspace(tmp_path, passed=False)
    ledger = tmp_path / "supertonic3.json"
    with pytest.raises(ValueError, match="override_reason"):
        record(converted, tarball, "v2.0.0", ledger)
    assert load_ledger(ledger)["entries"] == []


def test_record_with_override_is_loud_and_recorded(tmp_path):
    converted, tarball = _workspace(tmp_path, passed=False)
    ledger = tmp_path / "supertonic3.json"
    record(
        converted,
        tarball,
        "v2.0.0",
        ledger,
        override_reason="waived: threshold miscalibrated",
    )
    (entry,) = load_ledger(ledger)["entries"]
    assert entry["override_reason"] == "waived: threshold miscalibrated"

    passing_converted, passing_tarball = _workspace(tmp_path / "passing")
    with pytest.raises(ValueError, match="only applies"):
        record(
            passing_converted,
            passing_tarball,
            "v2.0.1",
            tmp_path / "other.json",
            override_reason="no reason to be here",
        )
