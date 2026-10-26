import json
from hashlib import sha256
from pathlib import Path

import pytest

from versta.batch import __main__ as batch
from versta.batch.publish_gate import PublishGateError
from versta.probe.bands import derive_bands, write_bands
from versta.schema import (
    validate_bundle_archive,
    validate_catalog,
    validate_generated_catalog,
)

from conftest import REGISTRY_URL
from test_bands import _card, _direction


def _input_catalog(pairs) -> list:
    """App-shaped input catalog entries, as the checked-in models.json carries."""
    return [
        [
            {
                "base_model": f"Firefox/{source}-{target}",
                "source_language": source,
                "target_language": target,
                "bidirectional": bidirectional,
                "architectures": ["BergamotTinyModel"],
                "score": 0.0,
                "version": "v0.0.0",
                "size": 1,
                "bundle": "https://models.versta.app/translation/pending.tar.gz",
                "checksum": "https://models.versta.app/translation/pending.tar.sha256",
            }
            for source, target in group
        ]
        for group, bidirectional in pairs
    ]


def _scorecards_and_bands_for(bundle: Path, tmp_path: Path):
    """
    Writes a passing pack-acceptance scorecard covering `bundle`'s exact
    bytes (the versta.probe output in the real flow) plus the golden bands
    derived from it, into tmp_path/scorecards and tmp_path/bands.json.
    """
    scorecards_dir = tmp_path / "scorecards"
    scorecards_dir.mkdir(exist_ok=True)
    card = _card("en-es", [_direction("en-es"), _direction("es-en", 82.9, 37.3)])
    card["bundle"]["file"] = bundle.name
    card["bundle"]["sha256"] = sha256(bundle.read_bytes()).hexdigest()
    (scorecards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")
    bands_file = tmp_path / "bands.json"
    write_bands(derive_bands(scorecards_dir), bands_file)
    return scorecards_dir, bands_file


def _export_kwargs(tmp_path: Path) -> dict:
    return {
        "input_file": tmp_path / "models.json",
        "output_dir": tmp_path / "export",
        "link_prefix": "https://models.versta.app/translation/",
        "registry_url": REGISTRY_URL,
    }


def test_batch_export_end_to_end_offline(mock_registry: dict, tmp_path: Path) -> None:
    """
    Smoke export: registry lookup -> download->verify -> bundle -> publish
    gate -> catalog sync, driven entirely against the mocked registry. Follows
    the operator flow for a newly built bundle: a dry-run export produces the
    bytes, the probe scorecard for those bytes is landed, and only then does
    the gated export publish.
    """
    input_file = tmp_path / "models.json"
    input_file.write_text(
        json.dumps(_input_catalog([([("en", "es"), ("es", "en")], True)]), indent=4)
    )

    # Step 1: dry-run export builds the bundle (bundle bytes are deterministic).
    batch.main(**_export_kwargs(tmp_path), skip_publish_gate=True)
    bundle = tmp_path / "export" / "en-es-bundle.tar.gz"
    assert bundle.is_file()

    # Step 2: probe measured, scorecard landed, bands pinned. Then the gated
    # run re-produces identical bytes and publishes.
    scorecards_dir, bands_file = _scorecards_and_bands_for(bundle, tmp_path)
    batch.main(
        **_export_kwargs(tmp_path), scorecards_dir=scorecards_dir, bands_file=bands_file
    )

    checksum = tmp_path / "export" / "en-es-bundle.tar.sha256"
    assert bundle.is_file()
    assert checksum.is_file()

    import hashlib

    assert checksum.read_text() == hashlib.sha256(bundle.read_bytes()).hexdigest()

    # The tarball is contract-valid.
    validate_bundle_archive(bundle)

    # The generated catalog and the synced input catalog are contract-valid.
    generated = json.loads((tmp_path / "export" / "models.json").read_text())
    validate_generated_catalog(generated)

    synced = json.loads(input_file.read_text())
    validate_catalog(synced)
    entry = synced[0][0]
    assert entry["version"] == "v2.1.0"
    assert entry["score"] == 87.4
    assert entry["size"] == bundle.stat().st_size
    assert entry["bundle"].endswith("en-es-bundle.tar.gz")

    # The batch input was backed up before syncing.
    assert (tmp_path / "models.json.bak").is_file()


def _bands_only(tmp_path: Path) -> Path:
    """Writes a synthetic bands document without any scorecard for the bundle
    under test — the gate needs the bands file to fail closed on cards."""
    cards_dir = tmp_path / "bands-src"
    cards_dir.mkdir()
    card = _card("en-es", [_direction("en-es"), _direction("es-en")])
    (cards_dir / "en-es.json").write_text(json.dumps(card, indent=2) + "\n")
    bands_file = tmp_path / "synthetic-bands.json"
    write_bands(derive_bands(cards_dir), bands_file)
    return bands_file


def test_batch_export_refuses_unproven_bundle(
    mock_registry: dict, tmp_path: Path
) -> None:
    """
    The NEU-60 core guarantee: a bundle with no passing scorecard cannot
    publish — the export produces the bytes, the gate refuses, and no
    catalog artifact is written or synced.
    """
    input_file = tmp_path / "models.json"
    input = _input_catalog([([("en", "es"), ("es", "en")], True)])
    input_file.write_text(json.dumps(input, indent=4))

    with pytest.raises(PublishGateError, match="no passing scorecard"):
        batch.main(
            **_export_kwargs(tmp_path),
            scorecards_dir=tmp_path / "empty-scorecards",
            bands_file=_bands_only(tmp_path),
        )

    assert not (tmp_path / "export" / "models.json").exists()
    assert json.loads(input_file.read_text()) == input


def test_batch_export_refuses_failing_scorecard(
    mock_registry: dict, tmp_path: Path
) -> None:
    """
    A deliberately broken pack (digit loss, the NEU-30 class) is backed by a
    failing scorecard and must not publish even though it built fine.
    """
    input_file = tmp_path / "models.json"
    input = _input_catalog([([("en", "es"), ("es", "en")], True)])
    input_file.write_text(json.dumps(input, indent=4))

    batch.main(**_export_kwargs(tmp_path), skip_publish_gate=True)
    bundle = tmp_path / "export" / "en-es-bundle.tar.gz"
    scorecards_dir, bands_file = _scorecards_and_bands_for(bundle, tmp_path)

    broken = json.loads((scorecards_dir / "en-es.json").read_text())
    broken["directions"][0]["digit_preservation"] = {
        "passed": 38,
        "total": 40,
        "failing_ids": ["num-02", "time-04"],
    }
    (scorecards_dir / "en-es.json").write_text(json.dumps(broken, indent=2) + "\n")

    published_after_run1 = json.loads(input_file.read_text())
    with pytest.raises(PublishGateError, match="digit preservation 38/40"):
        batch.main(
            **_export_kwargs(tmp_path),
            scorecards_dir=scorecards_dir,
            bands_file=bands_file,
        )

    # The publish-time catalog sync never ran: the input catalog is untouched
    # by the refused run.
    assert json.loads(input_file.read_text()) == published_after_run1


def test_batch_export_is_reproducible(mock_registry: dict, tmp_path: Path) -> None:
    """Two clean exports of the same pair produce byte-identical bundles."""
    checksums = []

    for run in ("first", "second"):
        run_dir = tmp_path / run
        run_dir.mkdir()
        input_file = run_dir / "models.json"
        input_file.write_text(
            json.dumps(_input_catalog([([("en", "es"), ("es", "en")], True)]), indent=4)
        )

        batch.main(
            input_file=input_file,
            output_dir=run_dir / "export",
            link_prefix="https://models.versta.app/translation/",
            registry_url=REGISTRY_URL,
            skip_publish_gate=True,
        )

        checksums.append((run_dir / "export" / "en-es-bundle.tar.sha256").read_text())

    assert checksums[0] == checksums[1]
