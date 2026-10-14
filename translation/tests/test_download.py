import json
from pathlib import Path

import pytest

from versta.download.download import (
    download_model,
    get_best_entry,
    get_entry,
    get_pair_entries,
    normalize_language,
    validate_registry_entry,
)
from versta.schema import ValidationError, validate_direction_metadata

from conftest import (
    fake_model_payload,
    make_registry,
    make_registry_entry,
)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("en", "en"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("pt-BR", "pt"),
        ("zh", "zh"),
        ("zh-Hans", "zh"),
        ("zh-CN", "zh"),
        ("zh-Hant", "zh_hant"),
        ("zh-TW", "zh_hant"),
        ("zh_HK", "zh_hant"),
        ("zh-mo", "zh_hant"),
        ("no", "no"),
        ("nb", "nb"),
        ("nn", "nn"),
        ("hbs", "hbs"),
    ],
)
def test_normalize_language(given: str, expected: str) -> None:
    assert normalize_language(given) == expected


def test_get_entry_prefers_tiny_then_base_memory() -> None:
    registry = {"models": {"en-es": []}}
    for architecture in ("base", "base-memory", "tiny"):
        registry["models"]["en-es"].append(
            make_registry_entry("en", "es", architecture)
        )

    assert get_entry(registry, "en", "es", None)["architecture"] == "tiny"

    registry["models"]["en-es"] = [
        entry
        for entry in registry["models"]["en-es"]
        if entry["architecture"] != "tiny"
    ]
    assert get_best_entry(registry, "en", "es")["architecture"] == "base-memory"


def test_get_entry_unknown_architecture_lists_available() -> None:
    registry = make_registry([("en", "es")])

    with pytest.raises(ValueError, match="Available architectures: tiny"):
        get_entry(registry, "en", "es", "base")


def test_get_entry_unknown_pair_raises() -> None:
    registry = make_registry([("en", "es")])
    assert get_pair_entries(registry, "en", "zh-TW") == []

    with pytest.raises(ValueError, match="No models available for 'en-zh_hant'"):
        get_entry(registry, "en", "zh-TW")


def test_validate_registry_entry_rejects_missing_kinds() -> None:
    entry = make_registry_entry("en", "es")
    validate_registry_entry(entry)

    for kind in ("model", "lexicalShortlist"):
        broken = make_registry_entry("en", "es")
        del broken["files"][kind]
        with pytest.raises(ValueError, match=kind):
            validate_registry_entry(broken)

    broken = make_registry_entry("en", "es")
    del broken["files"]["srcVocab"]
    with pytest.raises(ValueError, match=r"vocab\|srcVocab"):
        validate_registry_entry(broken)


def test_download_model_writes_verified_files_and_metadata(
    mock_registry: dict, tmp_path: Path
) -> None:
    entry = get_entry(mock_registry, "en", "es", "tiny")
    output_dir = download_model("https://models.example/", entry, tmp_path / "en-es")

    for name in ("model.bin", "lex.bin", "vocab.spm", "vocab.trg.spm"):
        assert (output_dir / name).is_file()

    assert (output_dir / "model.bin").read_bytes() == fake_model_payload(
        "model", "en", "es"
    )
    assert (output_dir / "vocab.trg.spm").read_bytes() == fake_model_payload(
        "trgVocab", "en", "es"
    )

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["directory"] == "en-es"
    assert metadata["source_language"] == "en"
    assert metadata["target_language"] == "es"
    assert metadata["architecture"] == "tiny"
    assert metadata["base_model"] == "en-es:tiny"
    assert metadata["score"] == pytest.approx(0.874)
    assert metadata["version"] == "v1.12.14"
    assert metadata["files"]["model"] == "model.bin"
    assert metadata["files"]["target_vocabulary"] == "vocab.trg.spm"

    # The produced metadata satisfies the app's loader contract.
    validate_direction_metadata(metadata, base_dir=output_dir)


def test_download_model_two_vocab_requires_trgvocab(
    mock_registry: dict, tmp_path: Path
) -> None:
    """NEU-26: the mock model carries split encoder_Wemb/decoder_Wemb
    embeddings (the en-ja shape), so an entry without trgVocab must not
    become a bundle the leanmt loader would refuse."""
    entry = make_registry_entry("en", "ja")
    del entry["files"]["trgVocab"]

    with pytest.raises(ValidationError, match="target_vocabulary"):
        download_model("https://models.example/", entry, tmp_path / "en-ja")


def test_download_model_en_ja_decoder_layers(
    mock_registry: dict, tmp_path: Path
) -> None:
    """Regression pin for the NEU-26 heads-up: the current en-ja generation
    (remote-settings v2.3) carries dec-depth 4; the manifest must pass the
    upstream modelConfig value through verbatim."""
    entry = make_registry_entry("en", "ja")
    output_dir = download_model("https://models.example/", entry, tmp_path / "en-ja")

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["config"]["decoder_layers"] == 4
    assert metadata["files"]["vocabulary"] == "vocab.spm"
    assert metadata["files"]["target_vocabulary"] == "vocab.trg.spm"


def test_download_model_aborts_on_hash_mismatch(
    mock_registry: dict, tmp_path: Path
) -> None:
    entry = get_entry(mock_registry, "en", "es", "tiny")
    entry["files"]["model"]["uncompressedHash"] = "0" * 64

    with pytest.raises(ValueError, match="Hash mismatch for model.bin"):
        download_model("https://models.example/", entry, tmp_path / "en-es")


def test_download_model_fails_fast_on_incomplete_entry(tmp_path: Path) -> None:
    entry = make_registry_entry("en", "es")
    del entry["files"]["lexicalShortlist"]

    with pytest.raises(ValueError, match="lexicalShortlist"):
        download_model("https://models.example/", entry, tmp_path / "en-es")

    # Nothing was downloaded when validation failed.
    assert not (tmp_path / "en-es" / "model.bin").exists()
