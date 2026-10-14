import json
import os
import tarfile

from hashlib import sha256
from pathlib import Path

import pytest

from versta.bundle import __main__ as bundle
from versta.bundle.bundle_tar import bundle_files, create_checksum
from versta.bundle.language import extract_unique_languages, validate_translation_pairs
from versta.bundle.metadata import load_metadata
from versta.schema import ValidationError, validate_bundle_archive

from conftest import write_direction_dir


def _metadata(direction: str) -> dict:
    return {
        "directory": Path(direction),
        "source_language": direction.split("-")[0],
        "target_language": direction.split("-")[1],
        "architecture": "tiny",
        "base_model": f"{direction}:tiny",
        "score": 0.874,
        "version": "v1.12.14",
        "files": {
            "model": "model.bin",
            "vocabulary": "vocab.spm",
            "target_vocabulary": None,
            "shortlist": "lex.bin",
        },
    }


def test_validate_translation_pairs_accepts_mirrored_pair() -> None:
    validate_translation_pairs([_metadata("en-es"), _metadata("es-en")])


def test_validate_translation_pairs_rejects_missing_reverse() -> None:
    with pytest.raises(ValueError, match=r"Missing translation pairs: \['es-en'\]"):
        validate_translation_pairs([_metadata("en-es")])


def test_extract_unique_languages_sorted() -> None:
    assert extract_unique_languages([_metadata("es-en"), _metadata("en-es")]) == [
        "en",
        "es",
    ]


def test_bundle_main_produces_deterministic_archive(tmp_path: Path) -> None:
    """Two bundles built from identical inputs are byte-identical."""
    checksums = []

    for run in ("run-a", "run-b"):
        work = tmp_path / run
        input_en_es = write_direction_dir(work / "input", "en-es")
        input_es_en = write_direction_dir(work / "input", "es-en")

        # Perturb file mtimes differently per run; outputs must not change.
        for index, file in enumerate(sorted(input_en_es.iterdir())):
            os.utime(file, (1_700_000_000 + index * 1000, 1_700_000_000 + index * 1000))

        result = bundle.main(
            input_dirs=[input_en_es, input_es_en],
            output_dir=work / "output",
            bidirectional=True,
            keep_intermediates=True,
            keep_input=True,
        )

        checksums.append(result["checksum"].read_text().strip())

        with tarfile.open(result["bundle"], "r:gz") as tar:
            names = tar.getnames()

        assert names == sorted(names)
        assert "metadata.json" in names
        assert "en-es/metadata.json" in names
        assert "es-en/model.bin" in names
        assert "en-es/vocab.trg.spm" in names

    assert checksums[0] == checksums[1]


def test_bundle_archive_validates_against_loader_contract(tmp_path: Path) -> None:
    input_en_es = write_direction_dir(tmp_path / "input", "en-es")
    input_es_en = write_direction_dir(tmp_path / "input", "es-en")

    result = bundle.main(
        input_dirs=[input_en_es, input_es_en],
        output_dir=tmp_path / "output",
        bidirectional=True,
    )

    # bundle.main already validates; this must not raise on the artifact.
    validate_bundle_archive(result["bundle"])

    with tarfile.open(result["bundle"], "r:gz") as tar:
        bundle_metadata = json.load(tar.extractfile("metadata.json"))

    assert bundle_metadata["version"] == "v2.0.0"
    assert bundle_metadata["bidirectional"] is True
    assert bundle_metadata["languages"] == ["en", "es"]
    assert {entry["directory"] for entry in bundle_metadata["metadata"]} == {
        "en-es",
        "es-en",
    }


def test_bundle_rejects_missing_reverse_pair(tmp_path: Path) -> None:
    input_en_es = write_direction_dir(tmp_path / "input", "en-es")

    with pytest.raises(ValueError, match="Missing translation pairs"):
        bundle.main(
            input_dirs=[input_en_es],
            output_dir=tmp_path / "output",
            bidirectional=True,
        )


def test_bundle_rejects_direction_with_missing_files(tmp_path: Path) -> None:
    input_en_es = write_direction_dir(tmp_path / "input", "en-es", missing=("model",))
    input_es_en = write_direction_dir(tmp_path / "input", "es-en")

    with pytest.raises(ValidationError, match="model.bin"):
        bundle.main(
            input_dirs=[input_en_es, input_es_en],
            output_dir=tmp_path / "output",
            bidirectional=True,
        )


def test_bundle_single_direction(tmp_path: Path) -> None:
    input_en_es = write_direction_dir(tmp_path / "input", "en-es")

    result = bundle.main(
        input_dirs=[input_en_es],
        output_dir=tmp_path / "output",
        bidirectional=False,
    )

    validate_bundle_archive(result["bundle"])
    with tarfile.open(result["bundle"], "r:gz") as tar:
        bundle_metadata = json.load(tar.extractfile("metadata.json"))
    assert bundle_metadata["bidirectional"] is False


def test_load_metadata_requires_all_fields(tmp_path: Path) -> None:
    folder = write_direction_dir(tmp_path, "en-es")
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["score"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="score"):
        load_metadata(folder)


def test_bundle_files_respects_source_date_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_direction_dir(tmp_path, "en-es")
    archive = tmp_path / "out.tar.gz"

    bundle_files([source], archive)
    with tarfile.open(archive, "r:gz") as tar:
        assert all(member.mtime == 0 for member in tar)

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    bundle_files([source], archive)
    with tarfile.open(archive, "r:gz") as tar:
        assert all(member.mtime == 1700000000 for member in tar)

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-a-number")
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        bundle_files([source], archive)


def test_create_checksum_matches_sha256(tmp_path: Path) -> None:
    payload = tmp_path / "artifact.bin"
    payload.write_bytes(b"versta")

    checksum = create_checksum(payload)
    assert checksum == tmp_path / "artifact.sha256"
    assert checksum.read_text() == sha256(b"versta").hexdigest()
