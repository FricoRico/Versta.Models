"""
Tests for the golden-band derivation/verification (NEU-60). Hermetic: cards
are written into tmp dirs; two repository-consistency tests additionally pin
that the committed bands.json still matches the committed scorecards/ — a
deliberate coupling: scorecard re-baselines must re-pin bands in the same
commit (NEU-16/22/23 discipline).
"""

import json
import subprocess
import sys

from pathlib import Path

import pytest

from versta.probe.bands import (
    BANDS_VERSION,
    DEFAULT_CHRF_MAX_DROP,
    BandsError,
    derive_bands,
    diff_bands,
    load_bands,
    write_bands,
)
from versta.probe.typing import BANDS_SCHEMA, SCORECARD_SCHEMA

TRANSLATION_ROOT = Path(__file__).resolve().parent.parent
SIGNATURE = "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0"


def _direction(key: str, chrf: float = 81.0, floor: float = 36.5) -> dict:
    return {
        "direction": key,
        "load_ok": True,
        "model_load_ms": 50.0,
        "rss_after_load_kib": 50000,
        "chrf": chrf,
        "chrf_floor": floor,
        "direction_check": "pass",
        "digit_preservation": {"passed": 40, "total": 40, "failing_ids": []},
        "latency_ms": {
            "n": 40,
            "min_ms": 1.0,
            "p50_ms": 2.0,
            "p95_ms": 3.0,
            "max_ms": 4.0,
        },
        "rss_peak_kib": 90000,
        "sentences": 40,
        "translation_failures": 0,
    }


def _card(
    slug: str,
    directions: list,
    acceptance: str = "pass",
    expectation: str = "pass",
    expectation_met: bool = True,
    corpus: str = "traveler-v1",
    signature: str = SIGNATURE,
) -> dict:
    return {
        "schema": SCORECARD_SCHEMA,
        "scorecard_version": 1,
        "generated_at": "2026-10-26T00:00:00+00:00",
        "slug": slug,
        "bundle": {
            "file": f"{slug}-bundle.tar.gz",
            "sha256": "ab" * 32,
            "leanmt_binary_version": 1,
            "bundle_version": "v2.1.0",
        },
        "environment": {
            "host_arch": "x86_64",
            "probe_arch": "x86_64",
            "threads": 2,
            "runner": "test",
            "leanmt_commit": "deadbeef",
            "corpus": corpus,
            "corpus_entries": 80,
            "corpus_sha256": "cd" * 32,
        },
        "metrics": {"sacrebleu_version": "2.6.0", "chrf_signature": signature},
        "directions": directions,
        "gates": {
            "load": True,
            "direction_chrf_floor": True,
            "digit_preservation": False,
            "latency_rss": False,
        },
        "acceptance": acceptance,
        "expectation": expectation,
        "expectation_met": expectation_met,
    }


def _write_cards(tmp_path: Path, cards: dict) -> Path:
    cards_dir = tmp_path / "scorecards"
    cards_dir.mkdir()
    for name, card in cards.items():
        (cards_dir / f"{name}.json").write_text(json.dumps(card, indent=2) + "\n")
    return cards_dir


def test_derive_bands_happy_path(tmp_path: Path) -> None:
    """A passing two-direction card yields two banded directions."""
    cards_dir = _write_cards(
        tmp_path,
        {
            "en-es": _card(
                "en-es", [_direction("en-es"), _direction("es-en", 82.9, 37.3)]
            )
        },
    )
    bands = derive_bands(cards_dir)

    assert bands["schema"] == BANDS_SCHEMA
    assert bands["bands_version"] == BANDS_VERSION
    assert bands["corpus"] == "traveler-v1"
    assert bands["chrf_max_drop"] == DEFAULT_CHRF_MAX_DROP
    assert bands["digit_preservation_required"] == 1.0
    assert bands["metrics"]["chrf_signature"] == SIGNATURE

    band = bands["directions"]["en-es"]
    assert band["golden_chrf"] == 81.0
    assert band["chrf_min"] == pytest.approx(81.0 - DEFAULT_CHRF_MAX_DROP)
    assert band["chrf_floor"] == 36.5
    assert band["baseline"]["scorecard"].endswith("en-es.json")
    assert len(bands["directions"]) == 2


def test_derive_skips_adversarial_and_aggregates(tmp_path: Path) -> None:
    """Negative controls and the sweep summary never anchor bands."""
    adversarial = _card(
        "en-es",
        [_direction("es-en", 22.3, 37.3)],
        acceptance="fail",
        expectation="direction-fail",
        expectation_met=True,
    )
    cards = {
        "en-es": _card("en-es", [_direction("en-es"), _direction("es-en")]),
        "es-en.adversarial": adversarial,
    }
    cards_dir = _write_cards(tmp_path, cards)
    (cards_dir / "sweep-summary.json").write_text(
        json.dumps({"schema": "versta.pack-sweep/1", "bundles": []}) + "\n"
    )
    bands = derive_bands(cards_dir)
    assert sorted(bands["directions"]) == ["en-es", "es-en"]
    assert bands["directions"]["es-en"]["golden_chrf"] == 81.0


def test_derive_refuses_failing_baseline(tmp_path: Path) -> None:
    """A failing baseline card is a hard error, not a band."""
    card = _card(
        "en-es", [_direction("en-es")], acceptance="fail", expectation_met=False
    )
    cards_dir = _write_cards(tmp_path, {"en-es": card})
    with pytest.raises(BandsError, match="not a passing card"):
        derive_bands(cards_dir)


def test_derive_refuses_band_below_sanity_floor(tmp_path: Path) -> None:
    """chrf_min must never undercut the direction-sanity floor."""
    card = _card("en-es", [_direction("en-es", chrf=38.0, floor=36.5)])
    cards_dir = _write_cards(tmp_path, {"en-es": card})
    with pytest.raises(BandsError, match="undercuts the direction-sanity floor"):
        derive_bands(cards_dir)


def test_derive_refuses_mixed_corpora(tmp_path: Path) -> None:
    """chrF is never comparable across corpora; banding them together is bogus."""
    cards_dir = _write_cards(
        tmp_path,
        {
            "en-es": _card("en-es", [_direction("en-es"), _direction("es-en")]),
            "en-fr": _card(
                "en-fr",
                [_direction("en-fr"), _direction("fr-en")],
                corpus="traveler-v2",
            ),
        },
    )
    with pytest.raises(BandsError, match="mix corpora"):
        derive_bands(cards_dir)


def test_diff_and_check_roundtrip(tmp_path: Path) -> None:
    """write_bands -> load_bands -> diff_bands(derived) is stable; a moved
    golden value is reported by --check semantics."""
    cards_dir = _write_cards(
        tmp_path, {"en-es": _card("en-es", [_direction("en-es"), _direction("es-en")])}
    )
    bands = derive_bands(cards_dir)
    bands_path = tmp_path / "pack-acceptance" / "bands.json"
    write_bands(bands, bands_path)

    pinned = load_bands(bands_path)
    assert diff_bands(derive_bands(cards_dir), pinned) == []

    pinned["directions"]["en-es"]["golden_chrf"] = 79.0
    write_bands(pinned, bands_path)
    drifted = diff_bands(derive_bands(cards_dir), load_bands(bands_path))
    assert any("en-es.golden_chrf" in line for line in drifted)

    with pytest.raises(BandsError, match="cannot load bands"):
        load_bands(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "versta.something-else/1"}))
    with pytest.raises(BandsError, match="expected schema"):
        load_bands(bad)


def test_committed_bands_match_committed_scorecards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Tree consistency lock: pack-acceptance/bands.json must equal what
    versta.probe.bands derives from the committed scorecards/ — both are
    pinned data, moved together in dedicated re-pin commits. Bands record
    scorecard provenance as repo-relative paths, so derive from the module
    root exactly like the documented re-pin command.
    """
    monkeypatch.chdir(TRANSLATION_ROOT)
    derived = derive_bands(Path("scorecards"))
    pinned = load_bands(Path("pack-acceptance") / "bands.json")
    diffs = diff_bands(derived, pinned)
    assert diffs == [], "bands drifted from scorecards: " + "; ".join(diffs)


def test_committed_bands_cover_every_published_direction() -> None:
    """The bands document pins exactly the 63 directions of the 32 bundles."""
    bands_file = TRANSLATION_ROOT / "pack-acceptance" / "bands.json"
    bands = load_bands(bands_file)
    assert len(bands["directions"]) == 63
    # Single-direction pack (NEU-19 verification target): only lv-en exists.
    assert "lv-en" in bands["directions"]
    assert "en-lv" not in bands["directions"]


def test_bands_cli_check(tmp_path: Path) -> None:
    """The module CLI exits 0 on a matching pin, 1 on drift."""
    cards_dir = _write_cards(
        tmp_path, {"en-es": _card("en-es", [_direction("en-es"), _direction("es-en")])}
    )
    bands_path = tmp_path / "bands.json"
    write_bands(derive_bands(cards_dir), bands_path)

    base_cmd = [
        sys.executable,
        "-m",
        "versta.probe.bands",
        "--scorecards_dir",
        str(cards_dir),
        "--output",
        str(bands_path),
    ]
    ok = subprocess.run([*base_cmd, "--check"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr

    moved = load_bands(bands_path)
    moved["directions"]["en-es"]["chrf_min"] = 70.0
    write_bands(moved, bands_path)
    drift = subprocess.run([*base_cmd, "--check"], capture_output=True, text=True)
    assert drift.returncode == 1
    assert "en-es.chrf_min" in drift.stdout
