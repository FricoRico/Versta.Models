import json
from pathlib import Path

from versta.batch import __main__ as batch
from versta.schema import (
    validate_bundle_archive,
    validate_catalog,
    validate_generated_catalog,
)

from conftest import REGISTRY_URL


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


def test_batch_export_end_to_end_offline(mock_registry: dict, tmp_path: Path) -> None:
    """
    Smoke export: registry lookup -> download->verify -> bundle -> catalog sync,
    driven entirely against the mocked registry. Asserts every produced artifact
    satisfies the app's loader contract.
    """
    input_file = tmp_path / "models.json"
    input_file.write_text(
        json.dumps(_input_catalog([([("en", "es"), ("es", "en")], True)]), indent=4)
    )

    batch.main(
        input_file=input_file,
        output_dir=tmp_path / "export",
        link_prefix="https://models.versta.app/translation/",
        registry_url=REGISTRY_URL,
    )

    bundle = tmp_path / "export" / "en-es-bundle.tar.gz"
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
    assert entry["version"] == "v2.0.0"
    assert entry["score"] == 87.4
    assert entry["size"] == bundle.stat().st_size
    assert entry["bundle"].endswith("en-es-bundle.tar.gz")

    # The batch input was backed up before syncing.
    assert (tmp_path / "models.json.bak").is_file()


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
        )

        checksums.append((run_dir / "export" / "en-es-bundle.tar.sha256").read_text())

    assert checksums[0] == checksums[1]
