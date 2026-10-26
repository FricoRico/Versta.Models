"""
Tests for the weekly-sweep regression evaluator (NEU-60): a sweep's
scorecards are judged against the pinned golden bands + publish rules and
condensed into the defect census the weekly workflow files from.
"""

import json

from pathlib import Path

from versta.probe.bands import derive_bands, write_bands
from versta.probe.regress import DEFECTS_SCHEMA, evaluate_sweep, write_defects

from test_bands import _card, _direction


def _sweep_env(tmp_path: Path, cards: dict, summary: dict = None):
    scorecards_dir = tmp_path / "sweep" / "scorecards"
    scorecards_dir.mkdir(parents=True)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    for name, card in cards.items():
        (scorecards_dir / f"{name}.json").write_text(json.dumps(card, indent=2) + "\n")
        # Negative controls are evaluated but never anchor baseline bands.
        if card["expectation"] != "pass":
            continue
        # The baseline cards derive bands; use clean full-digit versions.
        import copy

        baseline = copy.deepcopy(card)
        for direction in baseline["directions"]:
            direction["digit_preservation"] = {
                "passed": 40,
                "total": 40,
                "failing_ids": [],
            }
        (baseline_dir / f"{name}.json").write_text(
            json.dumps(baseline, indent=2) + "\n"
        )
    bands_file = tmp_path / "bands.json"
    write_bands(derive_bands(baseline_dir), bands_file)
    summary_path = None
    if summary is not None:
        summary_path = tmp_path / "sweep" / "sweep-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return scorecards_dir, bands_file, summary_path


def test_sweep_without_regressions_yields_empty_census(tmp_path: Path) -> None:
    cards = {"en-es": _card("en-es", [_direction("en-es"), _direction("es-en")])}
    scorecards_dir, bands_file, _ = _sweep_env(tmp_path, cards)
    census = evaluate_sweep(scorecards_dir, bands_file)

    assert census["schema"] == DEFECTS_SCHEMA
    assert census["cards_checked"] == 1
    assert census["defects_total"] == 0

    output = tmp_path / "defects.json"
    write_defects(census, output)
    assert json.loads(output.read_text())["defects_total"] == 0


def test_digit_and_band_regressions_become_defects(tmp_path: Path) -> None:
    clean_fr = _card(
        "en-fr", [_direction("en-fr", 80.0, 36.0), _direction("fr-en", 84.0, 38.0)]
    )
    cards = {
        "en-es": _card("en-es", [_direction("en-es"), _direction("es-en")]),
        "en-fr": clean_fr,
        "es-en.adversarial": _card(
            "en-es",
            [_direction("es-en", 22.3, 37.3)],
            acceptance="fail",
            expectation="direction-fail",
            expectation_met=True,
        ),
    }
    scorecards_dir, bands_file, _ = _sweep_env(tmp_path, cards)

    # The fresh sweep then observes the pack regressed below its baseline.
    degraded = json.loads((scorecards_dir / "en-fr.json").read_text())
    degraded["directions"][0]["digit_preservation"] = {
        "passed": 37,
        "total": 40,
        "failing_ids": ["num-02", "time-04", "price-01"],
    }
    degraded["directions"][1]["chrf"] = 70.0  # band breach (golden 84, min 81)
    (scorecards_dir / "en-fr.json").write_text(json.dumps(degraded, indent=2) + "\n")

    census = evaluate_sweep(scorecards_dir, bands_file)

    assert census["cards_checked"] == 2  # adversarial control is not counted
    assert census["defects_total"] == 1
    defect = census["defects"][0]
    assert defect["slug"] == "en-fr"
    blob = "\n".join(defect["failures"])
    assert "digit preservation 37/40" in blob
    assert "golden band" in blob


def test_hard_sweep_failure_is_a_defect(tmp_path: Path) -> None:
    """A bundle that errored before a card could be written still surfaces."""
    cards = {"en-es": _card("en-es", [_direction("en-es"), _direction("es-en")])}
    summary = {
        "schema": "versta.pack-sweep/1",
        "bundles_total": 2,
        "bundles_passing": 1,
        "bundles_hard_failing": 1,
        "bundles": [
            {
                "slug": "en-es",
                "acceptance": "pass",
                "expectation_met": True,
                "directions": [],
            },
            {
                "slug": "en-lv",
                "acceptance": "fail",
                "expectation_met": False,
                "error": "SweepError: checksum gate failed",
            },
        ],
    }
    scorecards_dir, bands_file, summary_path = _sweep_env(tmp_path, cards, summary)
    census = evaluate_sweep(scorecards_dir, bands_file, summary_path)

    assert census["defects_total"] == 1
    defect = census["defects"][0]
    assert defect["slug"] == "en-lv"
    assert "checksum gate failed" in defect["failures"][0]
