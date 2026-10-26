"""
Per-direction chrF floor calibration (NEU-59): floors come from the measured
kickoff sweep, never from static guesses — language distance varies too much
(a measured en-fr 87.4 vs en-zh 34.5 on the same corpus) for one number to
fit every pair.

Rule (deliberate, reviewable): ``floor = max(15.0, min(0.45 * chrf,
chrf - 20))`` — at least ~half of headroom below the measured value, and
always above the mislabel-collapse band (the adversarial es-en control
measures chrF 22.3 against its correct-direction references, so any floor at
or above ~25 still trips the NEU-19 class; the 15.0 minimum covers pairs
whose honest chrF sits below the collapse band of a same-script pair, where
a direction-mislabeled pack scores near zero anyway).
"""

import json

from pathlib import Path
from typing import Dict, List


def calibrate_floor(measured_chrf: float) -> float:
    """
    Computes the calibrated chrF floor for one direction from its measured
    first-sweep corpus chrF.

    Args:
        measured_chrf (float): corpus chrF measured on the real pack.

    Returns:
        float: the floor to record in the corpus file.
    """
    if measured_chrf <= 0.0:
        raise ValueError(f"cannot calibrate a floor from chrF {measured_chrf}")
    return max(
        15.0, min(round(0.45 * measured_chrf, 1), round(measured_chrf - 20.0, 1))
    )


def apply_calibration(corpus_dir: Path, sweep_summary_path: Path) -> Dict[str, float]:
    """
    Rewrites the ``chrf_floor`` (and calibration note) of every corpus under
    ``corpus_dir`` from a measured sweep summary (schema
    ``versta.pack-sweep/1``). Only directions that loaded cleanly calibrate —
    a pack that failed to load keeps its provisional floor so its defect
    report stays reproducible.

    Args:
        corpus_dir (Path): packaged probe corpora.
        sweep_summary_path (Path): sweep-summary.json from the kickoff sweep.

    Returns:
        Dict[str, float]: direction key -> new floor.

    Raises:
        ValueError: on malformed summary or a corpus/summary mismatch.
    """
    summary = json.load(open(sweep_summary_path, "r", encoding="utf-8"))
    if summary.get("schema") != "versta.pack-sweep/1":
        raise ValueError(
            f"{sweep_summary_path}: expected a versta.pack-sweep/1 summary"
        )

    measured: Dict[str, float] = {}
    for bundle in summary["bundles"]:
        for direction in bundle.get("directions", []):
            if direction["load_ok"]:
                measured[direction["direction"]] = float(direction["chrf"])

    updated: Dict[str, float] = {}
    calibrated: List[str] = []
    for corpus_path in sorted(corpus_dir.glob("*.json")):
        corpus = json.load(open(corpus_path, "r", encoding="utf-8"))
        direction_key = f"{corpus['source_language']}-{corpus['target_language']}"
        if direction_key not in measured:
            continue
        floor = calibrate_floor(measured[direction_key])
        corpus["chrf_floor"] = floor
        corpus["chrf_floor_note"] = (
            f"Calibrated from the NEU-59 kickoff sweep: measured corpus chrF "
            f"{measured[direction_key]:.1f} on the published v2.0.0 pack; floor = "
            "max(15.0, min(0.45 * chrF, chrF - 20)). A direction-mislabeled pack "
            "collapses far below this value (see the es-en adversarial control)."
        )
        corpus_path.write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        updated[direction_key] = floor
        calibrated.append(direction_key)
    if not updated:
        raise ValueError(
            f"no corpus under {corpus_dir} matched the sweep summary directions"
        )
    print(f"calibrated {len(updated)} direction floors:")
    for direction_key in calibrated:
        print(
            f"  {direction_key}: {measured[direction_key]:.1f} -> floor {updated[direction_key]}"
        )
    return updated


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate probe corpus chrF floors from a measured sweep summary."
    )
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--sweep_summary", type=Path, required=True)
    args = parser.parse_args()
    apply_calibration(args.corpus_dir, args.sweep_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
