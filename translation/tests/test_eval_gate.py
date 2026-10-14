"""Gate logic, ledger recording and the generated eval summary.

Hermetic (no network, no engine): predecessor and candidate are tiny
synthetic versta.eval-result/1 documents built by conftest.make_eval_result,
and ledgers live in tmp_path. Acceptance anchors: a regression fixture fails
the gate, an improvement passes, and the override path writes its reason
into the recorded ledger entry and the generated summary markdown.
"""

import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from versta.eval.evaluate import write_result
from versta.eval.ledger import (
    LedgerError,
    append_entry,
    empty_ledger,
    ledger_path,
    load_ledger,
    write_ledger,
)
from versta.eval.gate import GateError, evaluate_gate, gate_info, render_summary
from versta.eval.gate.__main__ import main as gate_main

from conftest import make_eval_result

NOW = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
LEDGER_REF = Path("evals/ledger/en-es.json")


def _pred_ledger(chrf: float = 60.0, version: str = "v1.12.14", **overrides):
    """An in-memory en-es ledger with one recorded predecessor version."""
    ledger = empty_ledger("en-es", "en", "es")
    result = make_eval_result(version=version, chrf=chrf, **overrides)
    report = evaluate_gate(
        result=result, ledger=ledger, ledger_path=LEDGER_REF, threshold=1.0, now=NOW
    )
    append_entry(ledger, result, gate_info(report))
    return ledger


def _gate(ledger, ledger_file: Optional[Path] = None, **kwargs):
    """Evaluates a candidate result against a ledger; kwargs go to evaluate_gate."""
    kwargs.setdefault("threshold", 1.0)
    return evaluate_gate(
        ledger=ledger, ledger_path=ledger_file or LEDGER_REF, now=NOW, **kwargs
    )


# --- gate decisions ---


def test_first_eval_is_baseline() -> None:
    ledger = empty_ledger("en-es", "en", "es")
    report = _gate(ledger, result=make_eval_result())

    assert report.decision == "baseline"
    assert report.ships
    assert report.predecessor is None
    assert report.delta_chrf is None and report.delta_bleu is None
    assert report.failure_why is None


def test_improvement_passes() -> None:
    ledger = _pred_ledger(chrf=60.0, bleu=41.0)
    report = _gate(ledger, result=make_eval_result(chrf=61.02, bleu=42.3))

    assert report.decision == "pass"
    assert report.ships
    assert report.predecessor["version"] == "v1.12.14"
    assert report.delta_chrf == pytest.approx(1.02)
    assert report.delta_bleu == pytest.approx(1.3)


def test_regression_within_threshold_passes() -> None:
    # Rule: fail on regression *greater than* the threshold; exactly -1.0 passes.
    ledger = _pred_ledger(chrf=60.0)
    assert _gate(ledger, result=make_eval_result(chrf=59.11)).ships
    assert _gate(ledger, result=make_eval_result(chrf=59.0)).decision == "pass"


def test_regression_beyond_threshold_fails() -> None:
    ledger = _pred_ledger(chrf=60.0)
    report = _gate(ledger, result=make_eval_result(chrf=58.4))

    assert report.decision == "fail"
    assert not report.ships
    assert report.delta_chrf == pytest.approx(-1.6)
    assert "chrF regressed by 1.6000 points vs. v1.12.14" in report.failure_why
    assert report.failure_why.endswith(
        "--reason (recorded in the ledger and the eval summary)."
    )


def test_threshold_is_configurable() -> None:
    ledger = _pred_ledger(chrf=60.0)
    assert (
        _gate(ledger, result=make_eval_result(chrf=59.2), threshold=0.5).decision
        == "fail"
    )
    assert (
        _gate(ledger, result=make_eval_result(chrf=59.2), threshold=1.0).decision
        == "pass"
    )


def test_override_ships_with_a_written_reason() -> None:
    ledger = _pred_ledger(chrf=60.0)
    report = _gate(
        ledger,
        result=make_eval_result(chrf=58.4),
        reason="forced upstream architecture switch tiny->base-memory (NEU-19): "
        "accepted -1.6 chrF for the footprint win; QA sign-off in release notes",
    )

    assert report.decision == "override"
    assert report.ships
    assert report.reason.startswith("forced upstream architecture switch")

    # The reason lands in the ledger record and in the summary markdown.
    info = gate_info(report)
    assert info["decision"] == "override"
    assert info["reason"] == report.reason
    assert info["predecessor_version"] == "v1.12.14"
    summary = render_summary(report)
    assert "**Gate: OVERRIDE**" in summary
    assert "**Override reason:** forced upstream architecture switch" in summary


def test_reason_on_a_clean_pass_is_rejected() -> None:
    ledger = _pred_ledger(chrf=60.0)
    with pytest.raises(GateError, match="nothing to justify"):
        _gate(ledger, result=make_eval_result(chrf=61.0), reason="unneeded")


def test_gate_requires_an_attributable_version_and_matching_direction() -> None:
    ledger = _pred_ledger()
    unversioned = make_eval_result()
    unversioned["model"]["version"] = ""
    with pytest.raises(GateError, match="no model version"):
        _gate(ledger, result=unversioned)

    with pytest.raises(GateError, match="es-en"):
        _gate(ledger, result=make_eval_result(direction="es-en"))


def test_rebaseline_requires_a_reason() -> None:
    ledger = _pred_ledger()
    with pytest.raises(GateError, match="rebaseline.*reason"):
        _gate(
            ledger,
            result=make_eval_result(corpus_revision="tests/fixture@0002"),
            rebaseline=True,
        )


def test_corpus_pin_change_fails_without_rebaseline() -> None:
    ledger = _pred_ledger()
    report = _gate(
        ledger, result=make_eval_result(corpus_revision="tests/fixture@0002")
    )

    assert report.decision == "fail"
    assert not report.ships
    assert "corpus revision changed" in report.failure_why
    assert "tests/fixture@0001" in report.failure_why
    # Deltas are still computed for the record, but the run is blocked.
    assert report.delta_chrf is not None


def test_corpus_pin_change_rebaseline_ships_with_reason() -> None:
    ledger = _pred_ledger()
    report = _gate(
        ledger,
        result=make_eval_result(corpus_revision="tests/fixture@0002", chrf=59.0),
        rebaseline=True,
        reason="corpus re-pinned to flores-200@abc123 after upstream fix (NEU-49)",
    )

    assert report.decision == "rebaseline"
    assert report.ships
    assert gate_info(report)["reason"].startswith("corpus re-pinned")
    summary = render_summary(report)
    assert "not comparable" in summary
    assert "tests/fixture@0002" in summary


# --- generated eval summary ---


def test_summary_carries_scores_deltas_and_size_notes() -> None:
    ledger = _pred_ledger(chrf=60.0, size_bytes=24_000_000)
    candidate = make_eval_result(chrf=61.0, size_bytes=23_500_000)
    summary = render_summary(_gate(ledger, result=candidate))

    assert summary.startswith("# Eval summary — en-es v2.0.0")
    assert "| chrF | 60.0000 | 61.0000 | +1.0000 |" in summary
    assert "| sacreBLEU | 41.2500 | 41.2500 | +0.0000 |" in summary
    assert "**Gate: PASS** (chrF regression threshold 1.0)" in summary
    assert "flores-200 dev @ `tests/fixture@0001`" in summary
    assert "bergamot-translator (local) bergamot-fake 0.0.0-test" in summary
    # Candidate is 500_000 bytes smaller than the recorded predecessor.
    assert "22.41 MiB (-500,000 B)" in summary


def test_summary_marks_unscored_comet_and_engine_change() -> None:
    ledger = _pred_ledger(engine_version="bergamot 0.9.0")
    report = _gate(ledger, result=make_eval_result(engine_version="bergamot 0.9.1"))
    summary = render_summary(report)

    assert "| COMET-22 | n/a | n/a | — |" in summary
    assert "previous: bergamot-translator (local) bergamot 0.9.0" in summary


# --- CLI: recording into the checked-in ledger, files written ---


def _cli_args(tmp_path: Path, result_path: Path, **extra):
    args = dict(
        result=result_path,
        ledger_dir=tmp_path / "evals" / "ledger",
        threshold=1.0,
        reason=None,
        rebaseline=False,
        record=True,
        summary_out=tmp_path / "summary.md",
    )
    args.update(extra)
    return args


def test_cli_records_pass_into_ledger_and_writes_summary(tmp_path: Path) -> None:
    ledger = _pred_ledger(chrf=60.0)
    write_ledger(ledger, ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    result_path = tmp_path / "candidate.json"
    write_result(make_eval_result(chrf=61.0), result_path)

    report = gate_main(**_cli_args(tmp_path, result_path))

    assert report.decision == "pass"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**Gate: PASS**" in summary

    recorded = load_ledger(ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    assert [entry["version"] for entry in recorded["entries"]] == ["v1.12.14", "v2.0.0"]
    entry = recorded["entries"][-1]
    assert entry["gate"]["decision"] == "pass"
    assert entry["gate"]["delta_chrf"] == pytest.approx(1.0)
    assert entry["result"]["corpus"]["revision"] == "tests/fixture@0001"


def test_cli_fail_records_nothing(tmp_path: Path) -> None:
    ledger = _pred_ledger(chrf=60.0)
    path = ledger_path(tmp_path / "evals" / "ledger", "en-es")
    write_ledger(ledger, path)
    result_path = tmp_path / "candidate.json"
    write_result(make_eval_result(chrf=58.0), result_path)

    report = gate_main(**_cli_args(tmp_path, result_path))

    assert report.decision == "fail"
    assert not report.ships
    # A failed gate never mutates history; the summary is still written as evidence.
    assert load_ledger(path)["entries"] == ledger["entries"]
    assert "**Blocked:**" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_cli_records_override_with_reason(tmp_path: Path) -> None:
    ledger = _pred_ledger(chrf=60.0)
    write_ledger(ledger, ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    result_path = tmp_path / "candidate.json"
    write_result(make_eval_result(chrf=58.0), result_path)
    reason = "accepted regression: upstream dropped tiny for this pair (NEU-19)"

    report = gate_main(**_cli_args(tmp_path, result_path, reason=reason))

    assert report.decision == "override"
    recorded = load_ledger(ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    entry = recorded["entries"][-1]
    assert entry["gate"]["decision"] == "override"
    assert entry["gate"]["reason"] == reason
    assert reason in (tmp_path / "summary.md").read_text(encoding="utf-8")

    # The recorded ledger still validates against the machine-checked contract.
    assert recorded["schema"] == "versta.eval-ledger/1"


def test_cli_record_refuses_a_duplicate_version(tmp_path: Path) -> None:
    ledger = _pred_ledger(chrf=60.0, version="v2.0.0")
    write_ledger(ledger, ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    result_path = tmp_path / "candidate.json"
    write_result(make_eval_result(chrf=99.0), result_path)

    # The candidate's own version is already recorded: the gate compares
    # against the *previous* version (none here), decides baseline, but the
    # append must refuse to fork history.
    with pytest.raises(LedgerError, match="already recorded"):
        gate_main(**_cli_args(tmp_path, result_path))


def test_cli_baseline_records_first_entry(tmp_path: Path) -> None:
    result_path = tmp_path / "candidate.json"
    write_result(make_eval_result(), result_path)

    report = gate_main(**_cli_args(tmp_path, result_path))

    assert report.decision == "baseline"
    recorded = load_ledger(ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    assert len(recorded["entries"]) == 1
    assert recorded["entries"][0]["gate"]["predecessor_version"] is None


# --- process-level behavior: exit codes the release job depends on ---


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "versta.eval.gate", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_exit_codes(tmp_path: Path) -> None:
    ledger = _pred_ledger(chrf=60.0)
    write_ledger(ledger, ledger_path(tmp_path / "evals" / "ledger", "en-es"))

    passing = tmp_path / "pass.json"
    write_result(make_eval_result(chrf=61.0), passing)
    failing = tmp_path / "fail.json"
    write_result(make_eval_result(chrf=50.0), failing)

    ledger_dir = tmp_path / "evals" / "ledger"
    ok = _run_cli(tmp_path, "--result", str(passing), "--ledger_dir", str(ledger_dir))
    assert ok.returncode == 0, ok.stderr
    assert "gate decision: pass" in ok.stdout

    bad = _run_cli(tmp_path, "--result", str(failing), "--ledger_dir", str(ledger_dir))
    assert bad.returncode == 1
    assert "gate decision: fail" in bad.stdout
    assert "chrF regressed" in bad.stdout
    # The failing run still wrote its eval summary as release-notes evidence.
    assert (tmp_path / "output" / "evals" / "en-es-v2.0.0-eval-summary.md").is_file()

    missing = _run_cli(tmp_path, "--result", str(tmp_path / "nope.json"))
    assert missing.returncode == 2
    assert "error:" in missing.stderr


def test_cli_override_via_process_exit_zero(tmp_path: Path) -> None:
    ledger = _pred_ledger(chrf=60.0)
    write_ledger(ledger, ledger_path(tmp_path / "evals" / "ledger", "en-es"))
    candidate = tmp_path / "candidate.json"
    write_result(make_eval_result(chrf=50.0), candidate)

    done = _run_cli(
        tmp_path,
        "--result",
        str(candidate),
        "--ledger_dir",
        str(tmp_path / "evals" / "ledger"),
        "--reason",
        "documented architecture downgrade accepted for size",
        "--record",
    )
    assert done.returncode == 0, done.stderr
    assert "gate decision: override (recorded in" in done.stdout
