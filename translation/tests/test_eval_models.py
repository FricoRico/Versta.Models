import tarfile

from pathlib import Path

import pytest

from versta.eval.models import (
    ModelInputError,
    load_direction_dir,
    load_from_bundle,
)
from versta.schema import ValidationError

from conftest import write_direction_dir, write_eval_bundle


def test_load_direction_dir_happy_path(tmp_path: Path) -> None:
    direction_dir = write_direction_dir(tmp_path, "en-es")
    model = load_direction_dir(direction_dir)

    assert model.direction == "en-es"
    assert model.base_model == "en-es:tiny"
    assert model.architecture == "tiny"
    assert model.model_path.is_file()
    assert model.shortlist_path.name == "lex.bin"
    assert model.target_vocabulary_path is not None
    assert len(model.model_sha256) == 64
    assert model.bundle_sha256 is None


def test_load_direction_dir_requires_directory(tmp_path: Path) -> None:
    with pytest.raises(ModelInputError, match="not found"):
        load_direction_dir(tmp_path / "en-es")


def test_load_direction_dir_enforces_app_schema(tmp_path: Path) -> None:
    direction_dir = write_direction_dir(tmp_path, "en-es")
    (direction_dir / "model.bin").unlink()
    with pytest.raises(ValidationError):
        load_direction_dir(direction_dir)


def test_load_from_bundle_happy_path(tmp_path: Path) -> None:
    bundle = write_eval_bundle(tmp_path)
    model = load_from_bundle(bundle, "en-es", tmp_path / "unpacked")

    assert model.direction == "en-es"
    assert model.bundle_file == bundle.name
    assert model.bundle_sha256 is not None
    assert model.model_path.is_file()
    assert (
        model.model_sha256
        == load_direction_dir(
            write_direction_dir(tmp_path / "again", "en-es")
        ).model_sha256
    )


def test_load_from_bundle_requires_bundle(tmp_path: Path) -> None:
    with pytest.raises(ModelInputError, match="not found"):
        load_from_bundle(tmp_path / "nope.tar.gz", "en-es", tmp_path / "u")


def test_load_from_bundle_rejects_unknown_direction(tmp_path: Path) -> None:
    bundle = write_eval_bundle(tmp_path)
    with pytest.raises(ModelInputError, match="carries no direction 'es-en'"):
        load_from_bundle(bundle, "es-en", tmp_path / "unpacked")


def test_load_from_bundle_rejects_missing_manifest(tmp_path: Path) -> None:
    direction_dir = write_direction_dir(tmp_path / "in", "en-es")
    archive_path = tmp_path / "no-manifest.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(direction_dir, arcname="en-es")
    with pytest.raises(ModelInputError, match="no bundle-level metadata.json"):
        load_from_bundle(archive_path, "en-es", tmp_path / "unpacked")


def test_load_from_bundle_rejects_unsafe_member(tmp_path: Path) -> None:
    bundle = write_eval_bundle(tmp_path)
    direction_dir = write_direction_dir(tmp_path / "evil-in", "en-es")

    # Graft a traversal member for the requested direction into an otherwise
    # valid archive; the harness must refuse before writing it to disk.
    poisoned = tmp_path / "poisoned.tar.gz"
    with (
        tarfile.open(bundle, "r:gz") as source,
        tarfile.open(poisoned, "w:gz") as target,
    ):
        for member in source:
            target.addfile(
                member,
                source.extractfile(member) if member.isfile() else None,
            )
        target.add(direction_dir / "metadata.json", arcname="en-es/../escape.json")

    with pytest.raises(ModelInputError, match="unsafe member path"):
        load_from_bundle(poisoned, "en-es", tmp_path / "unpacked")
    assert not (tmp_path / "unpacked" / "escape.json").exists()
