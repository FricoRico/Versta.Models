"""
Gate one candidate eval result against the direction's checked-in ledger:

    uv run python -m versta.eval.gate --result output/evals/en-es.json

    # Release path: record the decision into evals/ledger/<direction>.json
    uv run python -m versta.eval.gate --result output/evals/en-es.json --record

    # Ship a known regression, with the written reason recorded:
    uv run python -m versta.eval.gate --result output/evals/en-es.json \\
        --reason "upstream architecture switch tiny -> base-memory (NEU-19); \\
                  size halves, chrF -1.4 accepted for on-device footprint" \\
        --record

The gate is hermetic — it reads only the candidate's result JSON and the
direction ledger — so it runs identically on a laptop and in CI. It always
writes the generated release eval-summary markdown (default
output/evals/<direction>-<version>-eval-summary.md, override with
--summary_out) and prints it to stdout. A failed gate exits nonzero and is
never recorded; overrides and corpus rebaselines require a written
--reason that lands in the ledger entry and the summary.
"""

import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from ..config import DEFAULT_CHRF_REGRESSION_THRESHOLD
from ..evaluate import DEFAULT_OUTPUT_DIR, load_result
from ..ledger import (
    LedgerError,
    append_entry,
    empty_ledger,
    ledger_path,
    load_ledger,
    write_ledger,
)
from .check import GateError, GateReport, evaluate_gate, gate_info
from .summary import render_summary


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="""Release gate: compare one candidate eval result against the
        direction ledger (evals/ledger/<direction>.json) and decide whether
        the refresh ships. Always generates the release eval-summary
        markdown; --record appends a shipping decision to the checked-in
        ledger. Exit code 0 when the candidate ships, 1 on a failed gate,
        2 on invalid input or usage.""",
    )

    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="The candidate's versta.eval-result/1 JSON (written by `versta.eval`).",
    )

    parser.add_argument(
        "--ledger_dir",
        type=Path,
        default=Path("evals") / "ledger",
        help="Directory of per-direction ledgers (default: evals/ledger).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CHRF_REGRESSION_THRESHOLD,
        help="Maximum tolerated chrF regression in points vs. the predecessor "
        "(default: %(default)s, see versta/eval/config.py).",
    )

    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Written justification, recorded in the ledger entry and the "
        "eval summary. Required to ship a failing gate (override) or a "
        "corpus rebaseline; refused on a clean pass/baseline.",
    )

    parser.add_argument(
        "--rebaseline",
        action="store_true",
        default=False,
        help="Accept the candidate as a new baseline when the corpus pin "
        "changed vs. the predecessor (scores across pins are not "
        "comparable). Requires --reason.",
    )

    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Append the decision to the checked-in ledger "
        "(<ledger_dir>/<direction>.json). Refused on a failed gate.",
    )

    parser.add_argument(
        "--summary_out",
        type=Path,
        default=None,
        help="Eval-summary markdown destination (default: "
        "output/evals/<direction>-<version>-eval-summary.md).",
    )

    return parser.parse_args()


def main(
    result: Path,
    ledger_dir: Path,
    threshold: float,
    reason: Optional[str],
    rebaseline: bool,
    record: bool,
    summary_out: Optional[Path],
) -> GateReport:
    """
    Evaluates the gate, writes the eval summary, and optionally records the
    decision into the checked-in ledger.

    Args:
        result (Path): Candidate eval result JSON.
        ledger_dir (Path): Directory of per-direction ledgers.
        threshold (float): chrF regression threshold in points.
        reason (str | None): Written override/rebaseline justification.
        rebaseline (bool): Accept a corpus-pin change as a new baseline.
        record (bool): Append a shipping decision to the ledger.
        summary_out (Path | None): Summary markdown destination.

    Returns:
        GateReport: The gate evaluation (already rendered and, when
            requested and shipping, recorded).

    Raises:
        ValueError: If the result JSON is missing or malformed.
        LedgerError: If an existing ledger is malformed or the append would
            corrupt history.
        GateError: On meaningless argument combinations.
    """
    candidate = load_result(result)

    direction = candidate["direction"]
    path = ledger_path(ledger_dir, direction)
    if path.is_file():
        ledger = load_ledger(path)
    else:
        ledger = empty_ledger(
            direction,
            candidate["source_language"],
            candidate["target_language"],
        )

    report = evaluate_gate(
        result=candidate,
        ledger=ledger,
        ledger_path=path,
        threshold=threshold,
        reason=reason,
        rebaseline=rebaseline,
    )

    summary = render_summary(report)
    summary_path = summary_out or (
        DEFAULT_OUTPUT_DIR / f"{report.direction}-{report.version}-eval-summary.md"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")

    recorded = False
    if record:
        if report.ships:
            write_ledger(append_entry(ledger, candidate, gate_info(report)), path)
            recorded = True
        else:
            print(
                f"gate failed; nothing recorded in {path}",
                file=sys.stderr,
            )

    print(summary)
    print(
        f"gate decision: {report.decision}"
        + (f" (recorded in {path})" if recorded else "")
        + f" — summary: {summary_path}"
    )
    return report


if __name__ == "__main__":
    args = parse_args()
    try:
        gate_report = main(
            result=args.result,
            ledger_dir=args.ledger_dir,
            threshold=args.threshold,
            reason=args.reason,
            rebaseline=args.rebaseline,
            record=args.record,
            summary_out=args.summary_out,
        )
    except (ValueError, LedgerError, GateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if gate_report.ships else 1)
