import json
import struct
import tarfile
from pathlib import Path

import pytest

from versta.schema import (
    ValidationError,
    validate_bundle_metadata,
    validate_catalog,
    validate_catalog_entry,
    validate_catalog_group,
    validate_direction_metadata,
    validate_generated_catalog,
)

from conftest import write_direction_dir


def _direction(**overrides) -> dict:
    data = {
        "directory": "en-es",
        "source_language": "en",
        "target_language": "es",
        "architecture": "tiny",
        "base_model": "en-es:tiny",
        "score": 0.874,
        "version": "v1.12.14",
        "files": {
            "model": "model.bin",
            "vocabulary": "vocab.spm",
            "target_vocabulary": "vocab.trg.spm",
            "shortlist": "lex.bin",
        },
        "config": {
            "encoder_layers": 6,
            "decoder_layers": 2,
            "ffn_depth": 2,
            "num_heads": 8,
            "split_mode": "sentence",
        },
    }
    data.update(overrides)
    return data


def _bundle_metadata(**overrides) -> dict:
    data = {
        "version": "v2.1.0",
        "leanmt_binary_version": 1,
        "languages": ["en", "es"],
        "bidirectional": True,
        "metadata": [
            {"directory": "en-es", "source_language": "en", "target_language": "es"},
            {"directory": "es-en", "source_language": "es", "target_language": "en"},
        ],
    }
    data.update(overrides)
    return data


def _catalog_entry(**overrides) -> dict:
    entry = {
        "base_model": "Firefox/en-es",
        "source_language": "en",
        "target_language": "es",
        "bidirectional": True,
        "architectures": ["BergamotTinyModel"],
        "score": 85.4,
        "size": 51493339,
        "version": "v2.0.0",
        "bundle": "https://models.versta.app/translation/en-es-bundle.tar.gz",
        "checksum": "https://models.versta.app/translation/en-es-bundle.tar.sha256",
    }
    entry.update(overrides)
    return entry


class TestDirectionMetadata:
    def test_accepts_canonical_document(self, tmp_path: Path) -> None:
        folder = write_direction_dir(tmp_path, "en-es")
        data = json.loads((folder / "metadata.json").read_text())
        validate_direction_metadata(data, base_dir=folder)

    @pytest.mark.parametrize(
        "missing", ["base_model", "source_language", "files", "config"]
    )
    def test_rejects_missing_required_fields(self, missing: str) -> None:
        data = _direction()
        del data[missing]
        with pytest.raises(ValidationError, match=missing):
            validate_direction_metadata(data)

    def test_rejects_blank_language(self) -> None:
        with pytest.raises(ValidationError, match="target_language"):
            validate_direction_metadata(_direction(target_language="  "))

    def test_rejects_non_positive_config_dims(self) -> None:
        data = _direction()
        data["config"]["encoder_layers"] = 0
        with pytest.raises(ValidationError, match="encoder_layers"):
            validate_direction_metadata(data)

    def test_rejects_missing_files_on_disk(self, tmp_path: Path) -> None:
        folder = write_direction_dir(tmp_path, "en-es", missing=("shortlist",))
        data = json.loads((folder / "metadata.json").read_text())
        with pytest.raises(ValidationError, match="lex.bin"):
            validate_direction_metadata(data, base_dir=folder)

    def test_optional_fields_may_be_absent(self) -> None:
        data = _direction()
        del data["version"]
        del data["score"]
        data["files"]["target_vocabulary"] = None
        validate_direction_metadata(data)


class TestBundleMetadata:
    def test_accepts_canonical_document(self) -> None:
        validate_bundle_metadata(_bundle_metadata())

    def test_rejects_empty_languages(self) -> None:
        with pytest.raises(ValidationError, match="languages"):
            validate_bundle_metadata(_bundle_metadata(languages=[]))

    def test_rejects_odd_language_count_when_bidirectional(self) -> None:
        with pytest.raises(ValidationError, match="even number"):
            validate_bundle_metadata(_bundle_metadata(languages=["en", "es", "de"]))

    def test_rejects_metadata_entry_without_directory(self) -> None:
        data = _bundle_metadata()
        del data["metadata"][0]["directory"]
        with pytest.raises(ValidationError, match="directory"):
            validate_bundle_metadata(data)

    def test_rejects_missing_leanmt_binary_version(self) -> None:
        data = _bundle_metadata()
        del data["leanmt_binary_version"]
        with pytest.raises(ValidationError, match="leanmt_binary_version"):
            validate_bundle_metadata(data)

    def test_rejects_non_integer_leanmt_binary_version(self) -> None:
        with pytest.raises(ValidationError, match="expected integer"):
            validate_bundle_metadata(_bundle_metadata(leanmt_binary_version="1"))

    def test_fails_closed_on_other_leanmt_binary_versions(self) -> None:
        with pytest.raises(
            ValidationError, match="unsupported leanmt binary version 2"
        ):
            validate_bundle_metadata(_bundle_metadata(leanmt_binary_version=2))


class TestCatalog:
    def test_accepts_app_conformant_catalog(self) -> None:
        validate_catalog(
            [
                [
                    _catalog_entry(),
                    _catalog_entry(source_language="es", target_language="en"),
                ]
            ]
        )

    def test_rejects_unknown_architecture(self) -> None:
        with pytest.raises(ValidationError, match="unknown architecture"):
            validate_catalog_entry(_catalog_entry(architectures=["MarianBigModel"]))

    def test_rejects_relative_bundle_uri(self) -> None:
        with pytest.raises(ValidationError, match="absolute URI"):
            validate_catalog_entry(_catalog_entry(bundle="en-es-bundle.tar.gz"))

    def test_rejects_unmirrored_bidirectional_group(self) -> None:
        with pytest.raises(ValidationError, match="mirrored"):
            validate_catalog_group(
                [_catalog_entry(), _catalog_entry(source_language="de")]
            )

    def test_single_direction_group_needs_no_reverse(self) -> None:
        validate_catalog_group([_catalog_entry(bidirectional=False)])

    def test_rejects_empty_catalog(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            validate_catalog([])

    def test_checked_in_catalog_is_valid(self) -> None:
        """The real, checked-in models.json must always satisfy the app contract."""
        catalog_path = Path(__file__).parent.parent / "models.json"
        validate_catalog(json.loads(catalog_path.read_text()))


class TestGeneratedCatalog:
    def test_accepts_generated_shape(self) -> None:
        validate_generated_catalog(
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
                            "size": 12345,
                            "bundle": "https://models.example/en-es-bundle.tar.gz",
                            "checksum": "https://models.example/en-es-bundle.tar.sha256",
                        }
                    ]
                ],
            }
        )

    def test_rejects_missing_size(self) -> None:
        entry = {
            "source_language": "en",
            "target_language": "es",
            "architecture": "tiny",
            "score": 0.874,
            "version": "v1.12.14",
            "bundle": "https://models.example/en-es-bundle.tar.gz",
            "checksum": "https://models.example/en-es-bundle.tar.sha256",
        }
        with pytest.raises(ValidationError, match="size"):
            validate_generated_catalog({"version": "v2.0.0", "models": [[entry]]})


class TestBundleArchive:
    @staticmethod
    def _write_bundle_archive(
        tmp_path: Path, embedded_version: int = 1, declared_version: int = 1
    ) -> Path:
        """Writes a structurally valid single-direction bundle archive whose
        model file embeds `embedded_version` while the manifest declares
        `declared_version`."""
        from versta.bundle.bundle_tar import bundle_files

        direction = write_direction_dir(tmp_path / "input", "en-es")
        (direction / "model.bin").write_bytes(
            struct.pack("<Q", embedded_version) + b"payload:en-es:model"
        )

        manifest = _bundle_metadata(
            leanmt_binary_version=declared_version,
            bidirectional=False,
            metadata=[
                {"directory": "en-es", "source_language": "en", "target_language": "es"}
            ],
        )
        manifest_file = tmp_path / "metadata.json"
        manifest_file.write_text(json.dumps(manifest, indent=4))

        archive = tmp_path / "en-es-bundle.tar.gz"
        bundle_files([manifest_file, direction], archive)
        return archive

    def test_rejects_archive_without_bundle_metadata(self, tmp_path: Path) -> None:
        archive = tmp_path / "broken.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            payload = tmp_path / "readme.txt"
            payload.write_text("not a model bundle")
            tar.add(payload, arcname="readme.txt")

        from versta.schema import validate_bundle_archive

        with pytest.raises(ValidationError, match="metadata.json missing"):
            validate_bundle_archive(archive)

    def test_accepts_archive_with_matching_embedded_version(
        self, tmp_path: Path
    ) -> None:
        from versta.schema import validate_bundle_archive

        validate_bundle_archive(self._write_bundle_archive(tmp_path))

    def test_rejects_archive_with_mismatched_embedded_version(
        self, tmp_path: Path
    ) -> None:
        """A hand-edited manifest cannot smuggle an incompatible model past the
        gate: the archive check compares the payload header against the
        declared leanmt_binary_version."""
        from versta.schema import validate_bundle_archive

        archive = self._write_bundle_archive(
            tmp_path, embedded_version=2, declared_version=1
        )
        with pytest.raises(ValidationError, match="binary version mismatch"):
            validate_bundle_archive(archive)

    def test_rejects_archive_declaring_unsupported_version(
        self, tmp_path: Path
    ) -> None:
        from versta.schema import validate_bundle_archive

        archive = self._write_bundle_archive(
            tmp_path, embedded_version=2, declared_version=2
        )
        with pytest.raises(
            ValidationError, match="unsupported leanmt binary version 2"
        ):
            validate_bundle_archive(archive)
