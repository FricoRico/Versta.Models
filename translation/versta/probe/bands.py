"""
Golden-output bands (NEU-60): the pinned acceptable chrF band per shipped
direction, derived from the committed baseline scorecards and consumed by the
publish gate (``versta.batch``) and the weekly sweep.

A band anchors on the measurement taken on the previously published bundle
(us: the NEU-59 kickoff sweep cards in ``scorecards/``) and allows at most
``chrf_max_drop`` absolute chrF points of downward drift. The drop width is
deliberate: cross-runner wiggle between the x86_64 kickoff cards and the
arm64 CI runner has been measured at <=0.6 chrF points, while a real engine
or model regression moves several points and a direction-mislabeled pack
(NEU-19) collapses entirely. 3.0 sits between those regimes; widen or narrow
only via the re-pin procedure in PACK-ACCEPTANCE-RUNBOOK.md.

Re-pinning discipline is the one proven for bench baselines on NEU-16/22/23:
``python -m versta.probe.bands --check`` must stay green against the
committed scorecards, and any change to ``pack-acceptance/bands.json`` lands
as a dedicated commit with the measured scorecard diff attached and a git
note recording provenance. "Make CI green" is never a reason to move a band.
"""

import json

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from .sweep import SWEEP_SUMMARY_SCHEMA
from .typing import (
    BANDS_SCHEMA,
    SCORECARD_SCHEMA,
    BandBaseline,
    Bands,
    DirectionBand,
)

# Publish-allowed absolute chrF drop below the golden measurement. See the
# module docstring for the sizing rationale.
DEFAULT_CHRF_MAX_DROP = 3.0

BANDS_VERSION = 1

NOTE = (
    "Derived from the committed baseline scorecards by 'python -m "
    "versta.probe.bands'. Bands move only in dedicated re-pin commits with "
    "the measured scorecard diff attached (NEU-16/22/23 discipline); see "
    "PACK-ACCEPTANCE-RUNBOOK.md."
)


class BandsError(ValueError):
    """The bands document or its derivation inputs are malformed."""


def load_bands(bands_path: Path) -> Bands:
    """
    Loads and schema-checks the pinned golden-band document.

    Args:
        bands_path (Path): path to pack-acceptance/bands.json.

    Returns:
        Bands: the parsed bands.

    Raises:
        BandsError: missing/unreadable file or wrong schema/shape.
    """
    try:
        with open(bands_path, "r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BandsError(f"{bands_path}: cannot load bands: {error}")
    if not isinstance(data, dict) or data.get("schema") != BANDS_SCHEMA:
        seen = data.get("schema") if isinstance(data, dict) else type(data).__name__
        raise BandsError(
            f"{bands_path}: expected schema {BANDS_SCHEMA!r}, got {seen!r}"
        )
    for required in (
        "bands_version",
        "corpus",
        "chrf_max_drop",
        "digit_preservation_required",
        "metrics",
        "directions",
    ):
        if required not in data:
            raise BandsError(f"{bands_path}: missing {required!r}")
    if "chrf_signature" not in data["metrics"]:
        raise BandsError(f"{bands_path}: metrics lacks 'chrf_signature'")
    if not isinstance(data["directions"], dict) or not data["directions"]:
        raise BandsError(f"{bands_path}: 'directions' must be a non-empty map")
    return cast(Bands, data)


def _load_card(card_path: Path) -> Optional[Dict[str, Any]]:
    """
    Loads one scorecard, rejecting anything that is neither a v1 pack card
    nor the aggregate sweep summary that also lives under ``scorecards/``.
    Returns None for the sweep summary (an aggregate, not a baseline card).
    """
    try:
        with open(card_path, "r", encoding="utf-8") as handle:
            card: Any = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BandsError(f"{card_path}: cannot read scorecard: {error}")
    if isinstance(card, dict) and card.get("schema") == SWEEP_SUMMARY_SCHEMA:
        return None
    if not isinstance(card, dict) or card.get("schema") != SCORECARD_SCHEMA:
        raise BandsError(
            f"{card_path}: expected schema {SCORECARD_SCHEMA!r}; "
            "non-scorecard JSON in the scorecards directory"
        )
    return card


def derive_bands(
    scorecards_dir: Path, chrf_max_drop: float = DEFAULT_CHRF_MAX_DROP
) -> Bands:
    """
    Derives the golden-band document from the committed baseline scorecards.

    Every card must be a *passing* card under expectation ``pass`` ---
    adversarial controls are skipped, and a failing baseline card is a hard
    error: anchoring a band on a pack that already failed the v1 gates would
    bless a defect. Each direction records its measured chrF, the published
    chrf_min (``golden - chrf_max_drop``) and where the measurement came from.

    Args:
        scorecards_dir (Path): directory of committed scorecards.
        chrf_max_drop (float): allowed absolute chrF drop below golden.

    Returns:
        Bands: the derivable bands, directions sorted.

    Raises:
        BandsError: failing/loading-incomplete baselines, mixed corpora or
            mixed metric signatures across cards, or a chrf_min that would
            undercut the direction-sanity floor.
    """
    card_paths = sorted(scorecards_dir.glob("*.json"))
    if not card_paths:
        raise BandsError(f"{scorecards_dir}: no scorecards to derive bands from")

    directions: Dict[str, DirectionBand] = {}
    corpus_names = set()
    signatures = set()
    sacrebleu_versions = set()
    for card_path in card_paths:
        card = _load_card(card_path)
        if card is None:
            # The aggregate sweep summary lives next to the per-bundle cards;
            # it is not a baseline.
            continue
        if card["expectation"] != "pass":
            # Adversarial negative controls live next to the baselines; they
            # fail by design and must never anchor a band.
            continue
        if card["acceptance"] != "pass" or not card["expectation_met"]:
            raise BandsError(
                f"{card_path.name}: baseline card is not a passing card "
                f"(acceptance={card['acceptance']}, "
                f"expectation_met={card['expectation_met']}); "
                "fix the pack or the harness before anchoring bands on it"
            )
        environment = card["environment"]
        corpus_names.add(environment["corpus"])
        signatures.add(card["metrics"]["chrf_signature"])
        sacrebleu_versions.add(card["metrics"]["sacrebleu_version"])
        for direction in card["directions"]:
            key = direction["direction"]
            if not direction["load_ok"]:
                raise BandsError(
                    f"{card_path.name}: direction {key} did not load in the "
                    "baseline; cannot anchor a band on it"
                )
            golden = float(direction["chrf"])
            floor = float(direction["chrf_floor"])
            chrf_min = round(golden - chrf_max_drop, 2)
            if chrf_min < floor:
                raise BandsError(
                    f"{card_path.name}: direction {key}: band minimum "
                    f"{chrf_min} undercuts the direction-sanity floor "
                    f"{floor} (golden {golden}, drop {chrf_max_drop}) — "
                    "narrow the drop or re-check the baseline measurement"
                )
            if key in directions:
                raise BandsError(
                    f"direction {key} appears in more than one baseline card "
                    f"(at least {directions[key]['baseline']['scorecard']} and "
                    f"{card_path.name}); one bundle must own each direction"
                )
            directions[key] = DirectionBand(
                golden_chrf=golden,
                chrf_min=chrf_min,
                chrf_floor=floor,
                baseline=BandBaseline(
                    scorecard=(scorecards_dir / card_path.name).as_posix(),
                    bundle_sha256=card["bundle"]["sha256"],
                    leanmt_commit=environment.get("leanmt_commit"),
                    measured_on=f"{environment['runner']} ({environment['host_arch']})",
                ),
            )

    if len(corpus_names) != 1:
        raise BandsError(
            f"baseline cards mix corpora {sorted(corpus_names)}; chrF numbers "
            "are never comparable across corpora — re-baseline deliberately"
        )
    if len(signatures) != 1 or len(sacrebleu_versions) != 1:
        raise BandsError(
            f"baseline cards mix metric parameterizations "
            f"(signatures {sorted(signatures)}, sacrebleu "
            f"{sorted(sacrebleu_versions)}); bands would pin incomparable "
            "chrF number spaces"
        )

    return Bands(
        schema=BANDS_SCHEMA,
        bands_version=BANDS_VERSION,
        corpus=corpus_names.pop(),
        chrf_max_drop=chrf_max_drop,
        digit_preservation_required=1.0,
        metrics={
            "sacrebleu_version": sacrebleu_versions.pop(),
            "chrf_signature": signatures.pop(),
        },
        directions={key: directions[key] for key in sorted(directions)},
        note=NOTE,
    )


def write_bands(bands: Bands, output_path: Path) -> None:
    """Writes bands as stable, diff-friendly JSON (2-space indent)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bands, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def diff_bands(derived: Bands, pinned: Bands) -> List[str]:
    """
    Compares derived bands against the pinned document, returning one line
    per disagreement (empty list = byte-equivalent content). Used by
    ``--check`` to prove the committed bands still match the committed
    scorecards under the derivation rule.
    """
    diffs: List[str] = []
    full: Dict[str, Any] = {key: value for key, value in pinned.items()}
    if full.pop("schema", None) != derived["schema"]:
        diffs.append(f"schema: pinned {pinned.get('schema')!r}")
    for key in (
        "bands_version",
        "corpus",
        "chrf_max_drop",
        "digit_preservation_required",
    ):
        if pinned.get(key) != derived[key]:
            diffs.append(f"{key}: pinned {pinned.get(key)!r}, derived {derived[key]!r}")
    if pinned["metrics"] != derived["metrics"]:
        diffs.append(
            f"metrics: pinned {pinned['metrics']!r}, derived {derived['metrics']!r}"
        )

    pinned_directions = pinned["directions"]
    derived_directions = derived["directions"]
    for key in sorted(set(pinned_directions) - set(derived_directions)):
        diffs.append(f"direction {key}: pinned but not derivable from scorecards")
    for key in sorted(set(derived_directions) - set(pinned_directions)):
        diffs.append(f"direction {key}: derivable from scorecards but not pinned")
    for key in sorted(set(derived_directions) & set(pinned_directions)):
        old = pinned_directions[key]
        new = derived_directions[key]
        for field in ("golden_chrf", "chrf_min", "chrf_floor"):
            if old.get(field) != new.get(field):
                diffs.append(
                    f"direction {key}.{field}: pinned {old.get(field)!r}, "
                    f"derived {new.get(field)!r}"
                )
        if old.get("baseline") != new.get("baseline"):
            diffs.append(f"direction {key}.baseline: measurement provenance moved")
    return diffs


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Derive or verify the pinned golden-output bands "
        "(pack-acceptance/bands.json) from the committed baseline scorecards (NEU-60). "
        "Without --check, writes the derived document to --output (the re-pin "
        "step; land the change as a dedicated commit). With --check, exits 1 "
        "and lists the drift when the pinned bands no longer match the "
        "committed scorecards."
    )
    parser.add_argument(
        "--scorecards_dir",
        type=Path,
        default=Path("scorecards"),
        help="Directory of committed baseline scorecards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pack-acceptance/bands.json"),
        help="Bands document to write (or verify with --check).",
    )
    parser.add_argument(
        "--chrf_max_drop",
        type=float,
        default=DEFAULT_CHRF_MAX_DROP,
        help=f"Allowed absolute chrF drop below golden (default "
        f"{DEFAULT_CHRF_MAX_DROP}; change deliberately, it widens the gate).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify instead of writing: the pinned bands must equal the "
        "bands derived from --scorecards_dir.",
    )
    args = parser.parse_args()

    derived = derive_bands(args.scorecards_dir, chrf_max_drop=args.chrf_max_drop)
    if args.check:
        pinned = load_bands(args.output)
        diffs = diff_bands(derived, pinned)
        if diffs:
            print(
                f"bands check FAILED ({len(diffs)} drift(s)); re-pin "
                f"deliberately: python -m versta.probe.bands "
                f"--scorecards_dir {args.scorecards_dir} --output {args.output}"
            )
            for line in diffs:
                print(f"  {line}")
            return 1
        print(
            f"bands check OK: {len(pinned['directions'])} directions pinned, "
            f"corpus {pinned['corpus']}, max drop {pinned['chrf_max_drop']}"
        )
        return 0

    write_bands(derived, args.output)
    print(
        f"wrote {args.output}: {len(derived['directions'])} directions, "
        f"corpus {derived['corpus']}, max drop {derived['chrf_max_drop']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
