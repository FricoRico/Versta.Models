"""
Release-gate evaluation: candidate result vs. the ledger's predecessor.

The gate answers one question — *may this refresh ship?* — from two JSON
documents only (the candidate's ``versta.eval-result/1`` and the
direction's ``versta.eval-ledger/1``), so it is fully hermetic and cheap
enough for the PR path:

  * no predecessor → ``baseline`` (first recorded version ships the gates
    open);
  * corpus pin changed vs. the predecessor → scores are non-comparable by
    construction (see ``evals/README.md``), so the gate fails unless the
    refresh is deliberately accepted as a new baseline via ``--rebaseline``
    with a written ``--reason`` → decision ``rebaseline``;
  * chrF regression beyond the threshold (default
    :data:`versta.eval.config.DEFAULT_CHRF_REGRESSION_THRESHOLD`) → ``fail``,
    unless a written ``--reason`` is given → ``override``;
  * otherwise → ``pass``.

Decisions are data, not exceptions: a failing gate returns a report with
``decision == "fail"`` and a human-readable ``failure_why``; the CLI maps
that to a nonzero exit. Only genuinely invalid inputs (malformed result,
meaningless argument combinations) raise :class:`GateError`.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..ledger import predecessor_entry
from ..typing import DirectionLedger, EvalResult, GateInfo, LedgerEntry


class GateError(ValueError):
    """The gate cannot be evaluated: bad inputs or meaningless arguments."""


@dataclass(frozen=True)
class GateReport:
    """The outcome of one gate evaluation, inputs included (audit-friendly)."""

    result: EvalResult
    ledger: DirectionLedger
    ledger_path: Path
    decision: str
    ships: bool
    threshold_chrf: float
    predecessor: Optional[LedgerEntry]
    delta_bleu: Optional[float]
    delta_chrf: Optional[float]
    delta_comet22: Optional[float]
    reason: Optional[str]
    failure_why: Optional[str]
    decided_utc: str

    @property
    def direction(self) -> str:
        return self.result["direction"]

    @property
    def version(self) -> str:
        return self.result["model"]["version"]


def _delta(candidate: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Round(candidate − previous) at ledger precision; None when either side
    is unmeasured (e.g. COMET-22 skipped on one side)."""
    if candidate is None or previous is None:
        return None
    return round(candidate - previous, 4)


def evaluate_gate(
    *,
    result: EvalResult,
    ledger: DirectionLedger,
    ledger_path: Path,
    threshold: float,
    reason: Optional[str] = None,
    rebaseline: bool = False,
    now: Optional[datetime] = None,
) -> GateReport:
    """
    Evaluates the release gate for one candidate result against the ledger.

    Args:
        result (EvalResult): The candidate's eval result. Must carry a
            non-empty ``model.version`` — unattributable scores cannot ship.
        ledger (DirectionLedger): The direction's ledger *before* this
            candidate; must track the same direction.
        ledger_path (Path): Where the ledger lives (reported, not written).
        threshold (float): Maximum tolerated chrF regression in points
            (compared as ``delta < -threshold``).
        reason (str | None): Written justification; required to ship an
            override or a rebaseline, refused on a clean pass or baseline.
        rebaseline (bool): Accept a corpus-pin change as a new baseline
            (requires ``reason``).
        now (datetime | None): Decision timestamp (defaults to utcnow);
            injectable for tests.

    Returns:
        GateReport: The decision with deltas and provenance.

    Raises:
        GateError: On an unattributable result, a direction mismatch, or
            argument combinations that would make the record meaningless
            (reason on a clean pass, rebaseline without a reason).
    """
    version = str(result["model"].get("version", "")).strip()
    if not version:
        raise GateError(
            f"cannot gate {result.get('direction')}: the result carries no "
            f"model version — re-run versta.eval on an attributable artifact"
        )
    if result["direction"] != ledger["direction"]:
        raise GateError(
            f"result is for {result['direction']} but the ledger tracks "
            f"{ledger['direction']}"
        )
    if rebaseline and not (reason or "").strip():
        raise GateError("--rebaseline requires a written --reason")

    scores = result["scores"]
    decided = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    reason = reason.strip() if reason and reason.strip() else None
    predecessor = predecessor_entry(ledger, version)

    if predecessor is None:
        if reason is not None:
            raise GateError(
                "--reason records an override or a rebaseline; a first "
                "recorded version ships as baseline and takes no reason"
            )
        return GateReport(
            result=result,
            ledger=ledger,
            ledger_path=ledger_path,
            decision="baseline",
            ships=True,
            threshold_chrf=threshold,
            predecessor=None,
            delta_bleu=None,
            delta_chrf=None,
            delta_comet22=None,
            reason=None,
            failure_why=None,
            decided_utc=decided,
        )

    previous = predecessor["result"]["scores"]
    delta_chrf = _delta(scores["chrf"], previous["chrf"])
    delta_bleu = _delta(scores["bleu"], previous["bleu"])
    delta_comet22 = _delta(scores["comet22"], previous["comet22"])

    def report(decision: str, ships: bool, failure_why: Optional[str]) -> GateReport:
        return GateReport(
            result=result,
            ledger=ledger,
            ledger_path=ledger_path,
            decision=decision,
            ships=ships,
            threshold_chrf=threshold,
            predecessor=predecessor,
            delta_bleu=delta_bleu,
            delta_chrf=delta_chrf,
            delta_comet22=delta_comet22,
            reason=reason if decision in ("override", "rebaseline") else None,
            failure_why=failure_why,
            decided_utc=decided,
        )

    predecessor_revision = predecessor["result"]["corpus"]["revision"]
    if result["corpus"]["revision"] != predecessor_revision:
        if rebaseline:
            assert reason is not None  # enforced above
            return report("rebaseline", True, None)
        return report(
            "fail",
            False,
            f"corpus revision changed ({predecessor_revision} → "
            f"{result['corpus']['revision']}): scores are not comparable "
            f"across corpus pins. Re-run the gate with --rebaseline and a "
            f"written --reason to accept this candidate as the new baseline.",
        )

    assert delta_chrf is not None  # bleu/chrf are mandatory in both results
    if delta_chrf < -threshold:
        if reason is not None:
            return report("override", True, None)
        return report(
            "fail",
            False,
            f"chrF regressed by {-delta_chrf:.4f} points vs. {predecessor['version']} "
            f"(threshold {threshold}). Ship anyway by re-running with a "
            f"written --reason (recorded in the ledger and the eval summary).",
        )

    if reason is not None:
        raise GateError(
            "--reason records an override or a rebaseline; this candidate "
            f"passes the gate (chrF delta {delta_chrf:+.4f} within "
            f"threshold {threshold}), so there is nothing to justify"
        )
    return report("pass", True, None)


def gate_info(report: GateReport) -> GateInfo:
    """
    Builds the ledger's ``gate`` record from a gate report.

    Args:
        report (GateReport): A report whose decision ships (``baseline``,
            ``pass``, ``override`` or ``rebaseline``).

    Returns:
        GateInfo: The decision fields recorded next to the result in the
            direction ledger.

    Raises:
        GateError: If the report is a failed gate — a blocked refresh must
            never land in the ledger.
    """
    if not report.ships:
        raise GateError(
            f"cannot record a failed gate for {report.direction} "
            f"{report.version}: {report.failure_why}"
        )
    return GateInfo(
        decision=report.decision,
        threshold_chrf=report.threshold_chrf,
        predecessor_version=(
            report.predecessor["version"] if report.predecessor is not None else None
        ),
        delta_bleu=report.delta_bleu,
        delta_chrf=report.delta_chrf,
        delta_comet22=report.delta_comet22,
        reason=report.reason,
        decided_utc=report.decided_utc,
    )
