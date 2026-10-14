import json

from pathlib import Path

import pytest

from versta.eval.corpora import CorpusError, load_manifest, resolve_slice

from conftest import EVAL_PAIRS, write_eval_corpus


def test_load_manifest_happy_path(eval_corpus: Path) -> None:
    manifest = load_manifest(eval_corpus)
    assert manifest["corpus"] == "flores-200"
    assert manifest["split"] == "dev"
    assert "en-es" in manifest["directions"]


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_rejects_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "corpora.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="not valid JSON"):
        load_manifest(bad)


def test_load_manifest_rejects_wrong_schema(tmp_path: Path) -> None:
    manifest_path = write_eval_corpus(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["schema"] = "versta.eval-corpora/0"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CorpusError, match="expected schema"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_bad_digest(tmp_path: Path) -> None:
    manifest_path = write_eval_corpus(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["directions"]["en-es"]["source"]["sha256"] = "XYZ"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CorpusError, match="sha256"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_absolute_payload_path(tmp_path: Path) -> None:
    manifest_path = write_eval_corpus(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["directions"]["en-es"]["source"]["path"] = "/etc/passwd"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CorpusError, match="must be relative"):
        load_manifest(manifest_path)


def test_resolve_slice_reads_verified_sentences(eval_corpus: Path) -> None:
    manifest = load_manifest(eval_corpus)
    corpus = resolve_slice(manifest, "en-es", eval_corpus.parent / "corpora")
    assert corpus.source_sentences == [pair[0] for pair in EVAL_PAIRS]
    assert corpus.target_sentences == [pair[1] for pair in EVAL_PAIRS]
    assert len(corpus.source_sha256) == 64


def test_resolve_slice_unknown_direction_lists_available(
    eval_corpus: Path,
) -> None:
    manifest = load_manifest(eval_corpus)
    with pytest.raises(CorpusError, match="available: en-es"):
        resolve_slice(manifest, "de-fr", eval_corpus.parent / "corpora")


def test_resolve_slice_fails_on_payload_drift(eval_corpus: Path) -> None:
    manifest = load_manifest(eval_corpus)
    corpus_dir = eval_corpus.parent / "corpora"
    source_file = corpus_dir / manifest["directions"]["en-es"]["source"]["path"]
    source_file.write_text("tampered corpus line\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="hash mismatch"):
        resolve_slice(manifest, "en-es", corpus_dir)


def test_resolve_slice_fails_on_missing_payload(eval_corpus: Path) -> None:
    manifest = load_manifest(eval_corpus)
    corpus_dir = eval_corpus.parent / "corpora"
    (corpus_dir / manifest["directions"]["en-es"]["target"]["path"]).unlink()
    with pytest.raises(CorpusError, match="fetch the pinned corpus"):
        resolve_slice(manifest, "en-es", corpus_dir)


def test_resolve_slice_rejects_blank_source_lines(tmp_path: Path) -> None:
    pairs = [EVAL_PAIRS[0], ("", "Hueco."), *EVAL_PAIRS[1:]]
    manifest_path = write_eval_corpus(tmp_path, pairs=pairs)
    manifest = load_manifest(manifest_path)
    with pytest.raises(CorpusError, match="blank lines"):
        resolve_slice(manifest, "en-es", manifest_path.parent / "corpora")
