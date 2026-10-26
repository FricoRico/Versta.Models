"""
Publish gate (NEU-60): ``versta.batch`` refuses to publish a bundle into the
deployable catalog without a passing pack-acceptance scorecard.

A bundle is publishable only when a committed-style scorecard
(``versta.pack-scorecard/1``) exists for the exact tarball bytes produced
(joined on ``bundle.file`` + ``bundle.sha256``) and, per direction, all of
these hold:

* the pack loaded (``load_ok``);
* the direction sanity check passed (chrF above the corpus-carried floor —
  the NEU-19 mislabel tripwire);
* digit preservation is 100% (the NEU-30 harm class: corrupted numbers in
  dates, prices, times are silent user-facing damage);
* chrF sits inside the pinned golden-output band for the direction
  (``pack-acceptance/bands.json``): no regression against the previously
  published bundle beyond ``chrf_max_drop``.

The card must also have been produced under expectation ``pass`` (an
adversarial control card is never publish evidence), against the same
corpus and the same chrF parameterization the bands are pinned to.

Jointly owned with the pipeline pair per NEU-38: the scorecard/bands shapes
are manifest-adjacent — shape changes go through both pairs' review. The
operator re-pin procedure and failure taxonomy live in
``PACK-ACCEPTANCE-RUNBOOK.md``.
"""

import json

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..probe.bands import BandsError, load_bands
from ..probe.typing import SCORECARD_SCHEMA, Bands, DirectionBand

DEFAULT_SCORECARDS_DIR = Path("scorecards")
DEFAULT_BANDS_FILE = Path("pack-acceptance/bands.json")


class PublishGateError(RuntimeError):
    """
    One or more bundles may not be published. The message enumerates every
    failure across every bundle checked, so one run surfaces the full list.
    """


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_scorecard(
    bundle_name: str, bundle_sha256: str, scorecards_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Finds the unique scorecard that covers exactly these bundle bytes.

    Args:
        bundle_name (str): tarball file name (e.g. ``en-es-bundle.tar.gz``).
        bundle_sha256 (str): lowercase hex sha256 of the tarball.
        scorecards_dir (Path): directory of scorecard JSON files.

    Returns:
        Optional[Dict[str, Any]]: the matching card, or None.

    Raises:
        PublishGateError: more than one card claims the same bundle bytes
            (scorecard provenance is ambiguous — never guess).
    """
    matches: List[Dict[str, Any]] = []
    if scorecards_dir.is_dir():
        for card_path in sorted(scorecards_dir.glob("*.json")):
            try:
                with open(card_path, "r", encoding="utf-8") as handle:
                    card = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(card, dict) or card.get("schema") != SCORECARD_SCHEMA:
                continue
            bundle = card.get("bundle") or {}
            if (
                bundle.get("file") == bundle_name
                and bundle.get("sha256") == bundle_sha256
            ):
                card["_path"] = str(card_path)
                matches.append(card)
    if len(matches) > 1:
        paths = ", ".join(match["_path"] for match in matches)
        raise PublishGateError(
            f"{bundle_name}: {len(matches)} scorecards claim these bundle bytes "
            f"({paths}); keep exactly one card per bundle identity"
        )
    return matches[0] if matches else None


def check_direction(
    card_name: str, direction: Dict[str, Any], band: Optional[DirectionBand]
) -> List[str]:
    """
    Applies the publish rules to one direction of a scorecard against its
    pinned golden band. Returns a list of failure reasons (empty = pass).
    """
    failures: List[str] = []
    key = direction.get("direction", "?")
    prefix = f"{card_name}: direction {key}"

    if band is None:
        failures.append(
            f"{prefix}: no pinned golden band in bands.json — every shipped "
            "direction must be banded; re-pin via python -m versta.probe.bands"
        )
        return failures

    if not direction.get("load_ok", False):
        failures.append(
            f"{prefix}: pack does not load ({direction.get('load_error', 'unknown')})"
        )
        return failures

    if direction.get("direction_check") != "pass":
        failures.append(
            f"{prefix}: direction check failed — chrF {direction.get('chrf')} "
            f"below the direction-sanity floor {direction.get('chrf_floor')} "
            "(the NEU-19 mislabel tripwire)"
        )

    digits = direction.get("digit_preservation") or {}
    total = int(digits.get("total", 0))
    passed = int(digits.get("passed", 0))
    if total == 0 or passed != total:
        failing = ", ".join(digits.get("failing_ids", [])) or "none listed"
        failures.append(
            f"{prefix}: digit preservation {passed}/{total} "
            f"(failing: {failing}) — the gate requires 100% (NEU-30)"
        )

    chrf = float(direction.get("chrf", 0.0))
    if chrf < float(band["chrf_floor"]):
        failures.append(
            f"{prefix}: chrF {chrf} below the pinned sanity floor {band['chrf_floor']}"
        )
    if chrf < float(band["chrf_min"]):
        failures.append(
            f"{prefix}: chrF {chrf} below the golden band (golden "
            f"{band['golden_chrf']} on {band['baseline']['measured_on']}, "
            f"min {band['chrf_min']}) — a regression against the previously "
            "published bundle"
        )
    return failures


def check_card(card: Dict[str, Any], bundles_bands: Bands) -> List[str]:
    """
    Applies every card-level publish rule. Shared by the bundle gate and the
    weekly-sweep regression evaluator. Returns failure reasons (empty = the
    card backs a publishable bundle).
    """
    name = card.get("_path") or (card.get("bundle") or {}).get("file", "?")
    failures: List[str] = []

    if card.get("schema") != SCORECARD_SCHEMA:
        return [f"{name}: not a {SCORECARD_SCHEMA} scorecard"]

    if card.get("expectation") != "pass":
        failures.append(
            f"{name}: scorecard expectation is {card.get('expectation')!r} — "
            "adversarial/negative-control cards are never publish evidence"
        )
    if card.get("acceptance") != "pass" or not card.get("expectation_met"):
        failures.append(
            f"{name}: scorecard is not a passing card (acceptance="
            f"{card.get('acceptance')}, expectation_met="
            f"{card.get('expectation_met')})"
        )

    environment = card.get("environment") or {}
    if environment.get("corpus") != bundles_bands["corpus"]:
        failures.append(
            f"{name}: scored against corpus {environment.get('corpus')!r}, "
            f"bands are pinned to {bundles_bands['corpus']!r} — chrF numbers "
            "are never comparable across corpora; re-baseline deliberately"
        )
    signature = (card.get("metrics") or {}).get("chrf_signature")
    if signature != bundles_bands["metrics"]["chrf_signature"]:
        failures.append(
            f"{name}: chrF signature {signature!r} differs from the pinned "
            f"{bundles_bands['metrics']['chrf_signature']!r} — a metric or "
            "dependency bump is a deliberate re-baseline, not a silent upgrade"
        )

    for direction in card.get("directions", []):
        band = bundles_bands["directions"].get(direction.get("direction"))
        failures.extend(check_direction(name, direction, band))
    return failures


def check_bundle(
    bundle_path: Path, scorecards_dir: Path, bundles_bands: Bands
) -> List[str]:
    """
    Checks one produced bundle for publishability. Returns failure reasons
    (empty = publishable).
    """
    if not bundle_path.is_file():
        return [f"{bundle_path}: bundle tarball missing"]
    bundle_sha256 = _sha256_of(bundle_path)
    card = find_scorecard(bundle_path.name, bundle_sha256, scorecards_dir)
    if card is None:
        return [
            f"{bundle_path.name}: no passing scorecard covers these bundle "
            f"bytes (sha256 {bundle_sha256[:12]}…). Build, then probe the "
            "bundle (versta.probe) and land the card under scorecards/ "
            "before publishing — no bundle ships unproven."
        ]
    return check_card(card, bundles_bands)


def gate_bundles(
    bundle_paths: List[Path], scorecards_dir: Path, bands_file: Path
) -> None:
    """
    The gate ``versta.batch`` calls before writing the deployable catalog:
    refuses (raises PublishGateError) unless *every* bundle has a passing
    scorecard inside the pinned golden bands.

    Args:
        bundle_paths (List[Path]): freshly produced bundle tarballs.
        scorecards_dir (Path): directory of pack-acceptance scorecards.
        bands_file (Path): pinned golden bands (pack-acceptance/bands.json).

    Raises:
        PublishGateError: any bundle is unbacked, failing, or out of band;
            the message lists every reason across every bundle.
        BandsError: bands.json is missing/malformed — the gate fails closed
            when its own inputs are unreadable.
    """
    if not bands_file.is_file():
        raise BandsError(
            f"{bands_file}: pinned golden bands not found — the publish gate "
            "requires pack-acceptance/bands.json"
        )
    bundles_bands = load_bands(bands_file)

    failures: List[str] = []
    for bundle_path in sorted({Path(path) for path in bundle_paths}):
        failures.extend(check_bundle(bundle_path, scorecards_dir, bundles_bands))
    if failures:
        listing = "\n".join(f"  - {reason}" for reason in failures)
        raise PublishGateError(
            f"publish gate: refusing to publish "
            f"{len(bundle_paths)} bundle(s); {len(failures)} failure(s):\n{listing}"
        )


def main() -> int:
    """
    Standalone gate runner: checks bundles against scorecards + bands
    without running a batch export. Used by the operator flow ("can this
    bundle publish yet?") and by CI demonstrations (a deliberately broken
    pack must be refused).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="NEU-60 publish gate check: refuse to publish any bundle "
        "without a passing pack-acceptance scorecard inside the pinned "
        "golden bands. Exit 0 = every bundle publishable, 2 = refused."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        action="append",
        required=True,
        help="Bundle tarball to check; repeat for several bundles.",
    )
    parser.add_argument(
        "--scorecards_dir",
        type=Path,
        default=DEFAULT_SCORECARDS_DIR,
        help="Directory of pack-acceptance scorecards.",
    )
    parser.add_argument(
        "--bands_file",
        type=Path,
        default=DEFAULT_BANDS_FILE,
        help="Pinned golden bands document.",
    )
    args = parser.parse_args()

    bands = load_bands(args.bands_file)
    refused = 0
    for bundle_path in args.bundle:
        failures = check_bundle(bundle_path, args.scorecards_dir, bands)
        if failures:
            refused += 1
            print(f"REFUSE {bundle_path.name}:")
            for reason in failures:
                print(f"  - {reason}")
        else:
            print(f"PUBLISH-OK {bundle_path.name}")
    if refused:
        print(f"publish gate: {refused}/{len(args.bundle)} bundle(s) refused")
        return 2
    print(f"publish gate: all {len(args.bundle)} bundle(s) publishable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
