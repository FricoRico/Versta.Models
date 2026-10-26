"""
Tests for the full-catalog sweep driver and the kickoff-sweep chrF floor
calibration (NEU-59). Hermetic like test_probe.py: bundles come from the
production versta.bundle path over fake direction fixtures, and engine output
is canned — the real probe binary is exercised by the pack-acceptance CI
workflow.
"""

import io
import json
import tarfile

from pathlib import Path

import pytest

from conftest import write_direction_dir
from versta.bundle.__main__ import create_checksum
from versta.probe.calibrate import apply_calibration, calibrate_floor
from versta.probe.sweep import (
    SWEEP_SUMMARY_SCHEMA,
    BundleSweepResult,
    SweepReport,
    build_summary,
    repack_published_bundle,
    sweep_bundles,
    write_summary,
)


def _published_bundle(tmp_path: Path, directions: list, leanmt_field: bool) -> Path:
    """
    Builds a bundle tarball through the production bundler, then strips the
    NEU-25 handshake field from the manifest to mimic a pre-v2.1.0 published
    artifact like the live v2.0.0 packs.
    """
    input_dirs = [
        write_direction_dir(tmp_path / "downloads", direction)
        for direction in directions
    ]
    from versta.bundle.__main__ import main as bundle_main

    result = bundle_main(
        input_dirs=input_dirs,
        output_dir=tmp_path / "published",
        bidirectional=len(directions) > 1,
        keep_intermediates=False,
        keep_input=True,
    )
    bundle = result["bundle"]
    if not leanmt_field:
        # Rewrite the tarball minus leanmt_binary_version, as the v2.0.0
        # generation was published.
        source_names = bundle.name
        regenerated = bundle.parent / f"legacy-{source_names}"
        with tarfile.open(bundle, "r:gz") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
        manifest = json.loads(members["metadata.json"].decode("utf-8"))
        del manifest["leanmt_binary_version"]
        members["metadata.json"] = json.dumps(manifest, indent=4).encode("utf-8")
        with tarfile.open(regenerated, "w:gz") as tar:
            for name, payload in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        regenerated.rename(bundle)
        create_checksum(bundle)
    return bundle


def test_repack_restores_current_manifest(tmp_path):
    bundle = _published_bundle(tmp_path, ["en-es", "es-en"], leanmt_field=False)
    repacked = repack_published_bundle(bundle, tmp_path / "work", tmp_path / "repacked")
    with tarfile.open(repacked, "r:gz") as tar:
        manifest = json.loads(tar.extractfile("metadata.json").read().decode())
    assert manifest["leanmt_binary_version"] == 1  # NEU-25 handshake restored
    assert repacked.name == bundle.name


def test_sweep_collects_failures_instead_of_crashing(tmp_path):
    good = _published_bundle(tmp_path / "good", ["en-es", "es-en"], leanmt_field=True)
    # A truncated artifact never makes it past the checksum gate.
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    broken = broken_dir / "en-fr-bundle.tar.gz"
    broken.write_bytes(good.read_bytes() + b"junk")
    # Copy the real sidecar so the hash mismatch is the trap under test.
    sidecar = good.parent / (good.name[: -len(".gz")] + ".sha256")
    (broken_dir / "en-fr-bundle.tar.sha256").write_text(
        sidecar.read_text(encoding="utf-8")
    )

    report = sweep_bundles(
        tmp_path / "good" / "published",
        Path("/nonexistent/leanmt-probe"),
        tmp_path / "out",
        corpus_dir=tmp_path / "empty-corpora",
    )
    # No corpus files: every bundle errors, none crash the sweep.
    assert len(report.results) == 1
    assert report.results[0].error is not None

    summary = build_summary(report)
    assert summary["schema"] == SWEEP_SUMMARY_SCHEMA
    assert summary["bundles_total"] == 1
    assert summary["bundles_hard_failing"] == 1


def test_sweep_error_into_summary(tmp_path):
    report = SweepReport(
        results=[
            BundleSweepResult(
                slug="en-xx",
                published_file="en-xx-bundle.tar.gz",
                published_sha256="a" * 64,
                error="boom",
            )
        ]
    )
    summary = build_summary(report)
    assert summary["bundles"][0]["acceptance"] == "fail"
    assert summary["bundles"][0]["error"] == "boom"
    write_summary(report, tmp_path / "summary.json")
    reloaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert reloaded["schema"] == SWEEP_SUMMARY_SCHEMA


def test_calibrate_floor_rule():
    # Close-language pair with high measured chrF gets a floor near half the
    # measurement; the 0.45 factor dominates.
    assert calibrate_floor(80.0) == pytest.approx(36.0)
    # Far pair: chrF - 20 dominates, keeping the floor honest for low bands.
    assert calibrate_floor(40.0) == pytest.approx(18.0)
    # Very low measured chrF gets the hard minimum; a direction-mislabeled
    # pack collapses near zero there anyway.
    assert calibrate_floor(30.0) == pytest.approx(15.0)
    with pytest.raises(ValueError):
        calibrate_floor(0.0)


def _write_corpus(path: Path, source: str, target: str) -> None:
    entries = [
        {
            "id": f"t-{i}",
            "tags": ["numbers"],
            "source": f"Sentence {i} with digit {i % 10}.",
            "reference": f"Referenz {i} mit Ziffer {i % 10}.",
        }
        for i in range(40)
    ]
    path.write_text(
        json.dumps(
            {
                "name": "traveler-v1",
                "source_language": source,
                "target_language": target,
                "version": 1,
                "chrf_floor": 10.0,
                "chrf_floor_note": "provisional",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _write_summary(path: Path, directions: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "versta.pack-sweep/1",
                "bundles": [
                    {
                        "slug": "en-de",
                        "directions": [
                            {"direction": key, "load_ok": True, "chrf": value}
                            for key, value in directions.items()
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_apply_calibration_rewrites_floors(tmp_path):
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir / "en-de.json", "en", "de")
    _write_corpus(corpus_dir / "de-en.json", "de", "en")
    summary = tmp_path / "sweep-summary.json"
    _write_summary(summary, {"en-de": 78.8, "de-en": 83.0})

    updated = apply_calibration(corpus_dir, summary)
    assert updated == {"en-de": 35.5, "de-en": 37.4}

    corpus = json.loads((corpus_dir / "en-de.json").read_text(encoding="utf-8"))
    assert corpus["chrf_floor"] == pytest.approx(35.5)
    assert "kickoff sweep" in corpus["chrf_floor_note"]
    # Entries untouched — floors and the note are the only calibrated fields.
    assert len(corpus["entries"]) == 40


def test_apply_calibration_rejects_wrong_schema(tmp_path):
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    summary = tmp_path / "sweep.json"
    summary.write_text(json.dumps({"schema": "other/9", "bundles": []}))
    with pytest.raises(ValueError, match="versta.pack-sweep/1"):
        apply_calibration(corpus_dir, summary)
