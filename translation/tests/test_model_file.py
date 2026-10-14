import json
from pathlib import Path

import pytest

from versta.batch.model_file import (
    load_model_file,
    save_model_file,
    update_models_json,
)
from versta.schema import validate_catalog, validate_generated_catalog


def _write_input_catalog(path: Path, pairs=None) -> Path:
    """Writes an app-shaped (checked-in style) catalog for the given pairs."""
    pairs = pairs or [("en", "es"), ("es", "en")]
    document = [
        [
            {
                "base_model": f"Firefox/{source}-{target}",
                "source_language": source,
                "target_language": target,
                "bidirectional": True,
                "architectures": ["BergamotTinyModel"],
                "score": 0.0,
                "version": "v0.0.0",
                "size": 1,
                "bundle": "https://models.versta.app/translation/en-es-bundle.tar.gz",
                "checksum": "https://models.versta.app/translation/en-es-bundle.tar.sha256",
            }
            for source, target in pairs
        ]
    ]
    path.write_text(json.dumps(document, indent=4))
    return path


def _exported_bundle(source: str, target: str, bundle_path: Path, checksum_path: Path):
    return {
        "path": bundle_path,
        "checksum": checksum_path,
        "source_language": source,
        "target_language": target,
        "architecture": "tiny",
        "bidirectional": True,
        "score": 0.874,
        "version": "v1.12.14",
    }


class TestLoadModelFile:
    def test_parses_pairs_with_defaults(self, tmp_path: Path) -> None:
        catalog = tmp_path / "pairs.json"
        catalog.write_text(
            json.dumps(
                [
                    [
                        {"source_language": "en", "target_language": "es"},
                        {
                            "source_language": "es",
                            "target_language": "en",
                            "architecture": "base",
                            "score": 0.9,
                        },
                    ]
                ]
            )
        )

        models = load_model_file(catalog)

        assert len(models) == 1
        assert models[0][0]["architecture"] is None
        assert models[0][0]["score"] == 0.0
        assert models[0][1]["architecture"] == "base"
        assert models[0][1]["score"] == pytest.approx(0.9)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_file(tmp_path / "nope.json")


class TestSaveModelFile:
    def test_writes_valid_generated_catalog(self, tmp_path: Path) -> None:
        bundle_path = tmp_path / "en-es-bundle.tar.gz"
        bundle_path.write_bytes(b"bundle-bytes")
        checksum_path = tmp_path / "en-es-bundle.tar.sha256"
        checksum_path.write_text("deadbeef")

        bundles = [
            [
                _exported_bundle("en", "es", bundle_path, checksum_path),
                _exported_bundle("es", "en", bundle_path, checksum_path),
            ]
        ]

        output = save_model_file(bundles, "https://models.example/", tmp_path, "v2.0.0")

        generated = json.loads(output.read_text())
        validate_generated_catalog(generated)

        entry = generated["models"][0][0]
        assert generated["version"] == "v2.0.0"
        assert entry["size"] == len(b"bundle-bytes")
        assert entry["bundle"] == "https://models.example/en-es-bundle.tar.gz"
        assert entry["checksum"] == "https://models.example/en-es-bundle.tar.sha256"


class TestUpdateModelsJson:
    def _generated(self, tmp_path: Path) -> Path:
        generated_path = tmp_path / "generated" / "models.json"
        generated_path.parent.mkdir(exist_ok=True)
        generated_path.write_text(
            json.dumps(
                {
                    "version": "v2.0.0",
                    "models": [
                        [
                            {
                                "source_language": "en",
                                "target_language": "es",
                                "architecture": "tiny",
                                "score": 0.874,
                                "version": "v1.12.14",
                                "size": 999,
                                "bundle": "https://models.example/en-es-bundle.tar.gz",
                                "checksum": "https://models.example/en-es-bundle.tar.sha256",
                            },
                            {
                                "source_language": "es",
                                "target_language": "en",
                                "architecture": "tiny",
                                "score": 0.85,
                                "version": "v1.12.14",
                                "size": 999,
                                "bundle": "https://models.example/en-es-bundle.tar.gz",
                                "checksum": "https://models.example/en-es-bundle.tar.sha256",
                            },
                        ]
                    ],
                }
            )
        )
        return generated_path

    def test_syncs_computed_fields_and_scales_score(self, tmp_path: Path) -> None:
        catalog_path = _write_input_catalog(tmp_path / "models.json")

        result = update_models_json(catalog_path, self._generated(tmp_path), "v2.0.0")

        updated = json.loads(result.read_text())
        validate_catalog(updated)

        forward, reverse = updated[0]
        assert forward["version"] == "v2.0.0"
        assert forward["score"] == pytest.approx(87.4)
        assert reverse["score"] == pytest.approx(85.0)
        assert forward["size"] == 999
        assert forward["bundle"].endswith("en-es-bundle.tar.gz")
        # Descriptive fields are preserved.
        assert forward["base_model"] == "Firefox/en-es"
        assert forward["architectures"] == ["BergamotTinyModel"]

        # A backup was written alongside the updated catalog.
        assert (tmp_path / "models.json.bak").is_file()

    def test_bidirectional_reverse_lookup(self, tmp_path: Path) -> None:
        """Catalog entries match the generated reverse direction when their own
        direction was not regenerated."""
        catalog_path = _write_input_catalog(
            tmp_path / "models.json", [("en", "de"), ("de", "en")]
        )

        generated_path = tmp_path / "generated" / "models.json"
        generated_path.parent.mkdir(exist_ok=True)
        generated_path.write_text(
            json.dumps(
                {
                    "version": "v2.0.0",
                    "models": [
                        [
                            {
                                "source_language": "de",
                                "target_language": "en",
                                "architecture": "tiny",
                                "score": 0.881,
                                "version": "v1.12.14",
                                "size": 555,
                                "bundle": "https://models.example/de-en-bundle.tar.gz",
                                "checksum": "https://models.example/de-en-bundle.tar.sha256",
                            }
                        ]
                    ],
                }
            )
        )

        result = update_models_json(catalog_path, generated_path, "v2.0.0")

        updated = json.loads(result.read_text())
        forward, reverse = updated[0]
        assert forward["score"] == pytest.approx(88.1)
        assert forward["size"] == 555
        assert reverse["score"] == pytest.approx(88.1)
        validate_catalog(updated)
