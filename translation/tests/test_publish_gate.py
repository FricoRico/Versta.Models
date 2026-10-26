"""
Tests for the NEU-60 publish gate: versta.batch must refuse to publish any
bundle without a passing pack-acceptance scorecard for its exact bytes,
inside the pinned golden bands. Includes the deliberately-broken-pack
demonstration (missing card, failing card, out-of-band chrF) that CI runs.
"""

import json

from hashlib import sha256
from pathlib import Path

import pytest

from versta.batch.publish_gate import (
    PublishGateError,
    check_bundle,
    find_scorecard,
    gate_bundles,
)
from versta.probe.bands import derive_bands, write_bands

from test_bands import _card, _direction


def _bundle_file(
    tmp_path: Path, slug: str = "en-es", content: bytes = b"bundle-bytes"
) -> Path:
    bundle = tmp_path / f"{slug}-bundle.tar.gz"
    bundle.write_bytes(content)
    return bundle


def _publishable_env(tmp_path: Path, bundle: Path, chrf_a=81.0, chrf_b=82.9):
    """
    Writes a passing two-direction scorecard for `bundle` plus the bands
    derived from it; returns (scorecards_dir, bands_file).
    """
    scorecards_dir = tmp_path / "scorecards"
    scorecards_dir.mkdir(exist_ok=True)
    card = _card(
        "en-es",
        [_direction("en-es", chrf_a, 36.5), _direction("es-en", chrf_b, 37.3)],
    )
    card["bundle"]["file"] = bundle.name
    card["bundle"]["sha256"] = sha256(bundle.read_bytes()).hexdigest()
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")
    bands_file = tmp_path / "bands.json"
    write_bands(derive_bands(scorecards_dir), bands_file)
    return scorecards_dir, bands_file, card


def test_publish_gate_accepts_backed_bundle(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, _ = _publishable_env(tmp_path, bundle)
    gate_bundles([bundle], scorecards_dir, bands_file)  # no raise


def test_publish_gate_refuses_bundle_without_scorecard(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, _ = _publishable_env(tmp_path, bundle)
    # Tamper with the bytes after the card was written: nothing may publish
    # bytes a card does not cover.
    bundle.write_bytes(b"tampered bundle bytes")
    with pytest.raises(PublishGateError, match="no passing scorecard"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_when_no_scorecards_dir(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    _, bands_file, _ = _publishable_env(
        tmp_path, _bundle_file(tmp_path, content=b"other")
    )
    with pytest.raises(PublishGateError, match="no passing scorecard"):
        gate_bundles([bundle], tmp_path / "does-not-exist", bands_file)


def test_publish_gate_refuses_failing_scorecard(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["acceptance"] = "fail"
    card["expectation_met"] = False
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="not a passing card"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_adversarial_card_as_evidence(tmp_path: Path) -> None:
    """A negative-control card is a controlled failure, not publish evidence."""
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["expectation"] = "direction-fail"
    card["acceptance"] = "fail"
    card["expectation_met"] = True
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="never publish evidence"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_digit_loss(tmp_path: Path) -> None:
    """The NEU-30 class: 100% digit preservation is required to publish."""
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["directions"][0]["digit_preservation"] = {
        "passed": 39,
        "total": 40,
        "failing_ids": ["num-02"],
    }
    card["directions"][0]["chrf"] = 81.0
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="digit preservation 39/40"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_chrf_below_golden_band(tmp_path: Path) -> None:
    """A regression against the previously published bundle is refused even
    when the direction-sanity floor still passes."""
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    # Golden 81.0, drop 3.0 -> band min 78.0; floor 36.5 untouched.
    card["directions"][0]["chrf"] = 77.9
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="below the golden band"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_band_edge_inclusive(tmp_path: Path) -> None:
    """chrF exactly at chrf_min still passes the band."""
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["directions"][0]["chrf"] = 78.0  # == golden 81.0 - drop 3.0
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")
    gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_direction_floor_breach(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["directions"][0]["chrf"] = 30.0
    card["directions"][0]["direction_check"] = "fail"
    card["acceptance"] = "fail"
    card["expectation_met"] = False
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="direction check failed"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_corpus_or_metric_drift(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)

    drifted = dict(card)
    drifted["environment"] = {**card["environment"], "corpus": "traveler-v2"}
    (scorecards_dir / "en-es.json").write_text(json.dumps(drifted, indent=2) + "\n")
    with pytest.raises(PublishGateError, match="never comparable across corpora"):
        gate_bundles([bundle], scorecards_dir, bands_file)

    drifted = dict(card)
    drifted["metrics"] = {
        **card["metrics"],
        "chrf_signature": "nrefs:1|case:mixed|eff:yes|nc:8",
    }
    (scorecards_dir / "en-es.json").write_text(json.dumps(drifted, indent=2) + "\n")
    with pytest.raises(PublishGateError, match="deliberate re-baseline"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_unbanded_direction(tmp_path: Path) -> None:
    """Every shipped direction must be banded; an unknown direction refuses."""
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["directions"][1]["direction"] = "es-de"
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="no pinned golden band"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_publish_gate_refuses_load_failure(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["directions"][0] = {
        "direction": "en-es",
        "load_ok": False,
        "load_error": "leanmt: unsupported binary version",
        "direction_check": "fail",
    }
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    with pytest.raises(PublishGateError, match="does not load"):
        gate_bundles([bundle], scorecards_dir, bands_file)


def test_find_scorecard_rejects_ambiguity(tmp_path: Path) -> None:
    bundle = _bundle_file(tmp_path)
    scorecards_dir, _, card = _publishable_env(tmp_path, bundle)
    (scorecards_dir / "duplicate.json").write_text(json.dumps(card, indent=2) + "\n")
    with pytest.raises(PublishGateError, match="claim these bundle bytes"):
        find_scorecard(
            bundle.name, sha256(bundle.read_bytes()).hexdigest(), scorecards_dir
        )


def test_gate_requires_bands_file(tmp_path: Path) -> None:
    """The gate fails closed when its own pinned inputs are missing."""
    bundle = _bundle_file(tmp_path)
    with pytest.raises(Exception, match="golden bands not found"):
        gate_bundles([bundle], tmp_path, tmp_path / "missing-bands.json")


def test_check_bundle_reports_every_failure(tmp_path: Path) -> None:
    """All failure classes surface in one run, so one refusal is the full list."""
    bundle = _bundle_file(tmp_path)
    scorecards_dir, bands_file, card = _publishable_env(tmp_path, bundle)
    card["acceptance"] = "fail"
    card["expectation_met"] = False
    card["directions"][0]["chrf"] = 70.0
    card["directions"][1]["digit_preservation"] = {
        "passed": 30,
        "total": 40,
        "failing_ids": ["num-01"],
    }
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")

    from versta.probe.bands import load_bands

    failures = check_bundle(bundle, scorecards_dir, load_bands(bands_file))
    blob = "\n".join(failures)
    assert "not a passing card" in blob
    assert "golden band" in blob
    assert "digit preservation 30/40" in blob
