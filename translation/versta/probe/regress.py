"""
Weekly-sweep regression evaluation (NEU-60): scores a full sweep's
scorecards against the pinned golden bands and the publish rules, then
writes a machine-readable defect census (``versta.pack-defects/1``) the
weekly workflow turns into filed defect issues.

Two defect sources are unioned per bundle slug:

* hard sweep failures from the ``versta.pack-sweep/1`` summary (a bundle
  that errored or failed its expectation at probe time — e.g. the pack no
  longer installs, or a negative control stopped tripping);
* publish-rule failures from the individual cards
  (``versta.batch.publish_gate.check_card``): digit preservation below
  100%, direction/chrF sanity, or chrF outside the pinned golden band.

The evaluator never re-probes: it judges exactly the cards the sweep
emitted, so a defect links to reproducible evidence.
"""

import json

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..batch.publish_gate import check_card
from .bands import load_bands
from .sweep import SWEEP_SUMMARY_SCHEMA
from .typing import SCORECARD_SCHEMA

# Named schema for the defect census, written as defects.json next to the
# sweep summary and filed per bundle by the weekly workflow.
DEFECTS_SCHEMA = "versta.pack-defects/1"


class SweepRegressionError(RuntimeError):
    """The sweep outputs are missing/malformed; nothing can be evaluated."""


def _defect(
    slug: str, card: str, bundle_sha256: Optional[str], failures: List[str]
) -> Dict[str, Any]:
    return {
        "slug": slug,
        "card": card,
        "bundle_sha256": bundle_sha256,
        "failures": failures,
    }


def evaluate_sweep(
    scorecards_dir: Path, bands_file: Path, sweep_summary_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates one sweep run's scorecards against the bands/publish rules.

    Args:
        scorecards_dir (Path): sweep output scorecards (per-bundle cards;
            adversarial controls are controls, never defects).
        bands_file (Path): the pinned golden bands.
        sweep_summary_path (Optional[Path]): the sweep's summary; when given,
            bundles that hard-failed before a card could be written still
            appear as defects.

    Returns:
        Dict[str, Any]: the defect census document (schema
            ``versta.pack-defects/1``), defects sorted by slug.

    Raises:
        SweepRegressionError: directories/files missing or malformed.
    """
    if not scorecards_dir.is_dir():
        raise SweepRegressionError(f"{scorecards_dir}: no sweep scorecards directory")
    bands = load_bands(bands_file)

    defects: Dict[str, Dict[str, Any]] = {}
    cards_checked = 0
    for card_path in sorted(scorecards_dir.glob("*.json")):
        try:
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise SweepRegressionError(f"{card_path}: cannot read card: {error}")
        if not isinstance(card, dict) or card.get("schema") != SCORECARD_SCHEMA:
            # sweep-summary.json or foreign JSON: not a per-bundle card.
            continue
        if card.get("expectation") != "pass":
            # Negative controls validate the harness; their verdicts are
            # expected failures and are tracked by the sweep summary itself.
            continue
        cards_checked += 1
        failures = check_card(card, bands)
        if failures:
            slug = card.get("slug") or card_path.stem
            defects[slug] = _defect(
                slug,
                card_path.name,
                (card.get("bundle") or {}).get("sha256"),
                failures,
            )

    if sweep_summary_path is not None:
        with open(sweep_summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if (
            not isinstance(summary, dict)
            or summary.get("schema") != SWEEP_SUMMARY_SCHEMA
        ):
            raise SweepRegressionError(
                f"{sweep_summary_path}: expected a {SWEEP_SUMMARY_SCHEMA} summary"
            )
        for bundle in summary["bundles"]:
            if bundle.get("expectation_met"):
                continue
            slug = bundle["slug"]
            reasons = defects.get(
                slug, _defect(slug, "", bundle.get("repacked_sha256"), [])
            )
            reasons["failures"].append(
                "hard sweep failure: "
                + (bundle.get("error") or "scorecard expectation not met")
            )
            defects[slug] = reasons

    return {
        "schema": DEFECTS_SCHEMA,
        "corpus": bands["corpus"],
        "cards_checked": cards_checked,
        "defects_total": len(defects),
        "defects": [defects[slug] for slug in sorted(defects)],
    }


def write_defects(defects: Dict[str, Any], output_path: Path) -> None:
    """Writes the defect census as stable, diff-friendly JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(defects, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a pack-acceptance sweep against the pinned golden "
        "bands and publish rules; write the defect census (defects.json) the "
        "weekly workflow files as defects (NEU-60)."
    )
    parser.add_argument(
        "--scorecards_dir",
        type=Path,
        required=True,
        help="Sweep output scorecards directory.",
    )
    parser.add_argument(
        "--bands_file",
        type=Path,
        default=Path("pack-acceptance/bands.json"),
        help="Pinned golden bands.",
    )
    parser.add_argument(
        "--sweep_summary",
        type=Path,
        default=None,
        help="Optional versta.pack-sweep/1 summary; hard sweep failures become defects.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write defects.json (defaults to <scorecards_dir>/../defects.json).",
    )
    args = parser.parse_args()

    census = evaluate_sweep(args.scorecards_dir, args.bands_file, args.sweep_summary)
    output = args.output or (args.scorecards_dir.parent / "defects.json")
    write_defects(census, output)
    print(
        f"regression sweep: {census['cards_checked']} cards checked, "
        f"{census['defects_total']} defect(s) -> {output}"
    )
    for defect in census["defects"]:
        print(f"  DEFECT {defect['slug']} ({len(defect['failures'])} failure(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
