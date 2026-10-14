import hashlib
import io
import json
import tarfile

from pathlib import Path
from typing import Dict, Tuple

import pytest

from versta.eval.corpora.manifest import load_manifest
from versta.eval.corpora.slices import (
    CorpusError,
    load_manifest as load_harness_manifest,
)
from versta.eval.corpora.slices import resolve_slice, snapshot_dir

from conftest import EVAL_PAIRS, write_eval_corpus
from versta.eval.corpora.pipeline import (
    build_snapshot,
    derive_directions,
    fetch_snapshot,
)
from versta.eval.corpora.typing import CatalogEntry

LANGUAGES: Dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "lv": "lvs_Latn",
    "mt": "mlt_Latn",
}


def _sentences(code: str, count: int = 997) -> bytes:
    return "".join(f"{code} sentence {number}\n" for number in range(count)).encode(
        "utf-8"
    )


def make_upstream_tarball(
    tmp_path: Path,
    languages: Dict[str, str] = LANGUAGES,
    counts: Dict[str, int] | None = None,
) -> Tuple[Path, Dict[str, bytes]]:
    """Writes a minimal FLORES-style upstream archive with fake sentences."""
    payloads = {
        code: _sentences(code, (counts or {}).get(code, 997))
        for code in languages.values()
    }
    archive = tmp_path / "flores200_dataset.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for code, data in payloads.items():
            # Upstream member names carry a "./" prefix.
            info = tarfile.TarInfo(f"./flores200_dataset/dev/{code}.dev")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive, payloads


def make_catalog(tmp_path: Path) -> Path:
    """A two-group catalog: one mirrored pair and one single-direction entry
    whose fields are mislabeled (mirrors the shipped Firefox/mt-lv group)."""
    entry = {
        "base_model": "Firefox/en-es",
        "source_language": "en",
        "target_language": "es",
        "bidirectional": True,
    }
    mirrored = dict(
        entry, base_model="Firefox/es-en", source_language="es", target_language="en"
    )
    single = {
        "base_model": "Firefox/mt-lv",
        "source_language": "lv",
        "target_language": "en",
        "bidirectional": False,
    }
    catalog = tmp_path / "catalog" / "models.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(json.dumps([[entry, mirrored], [single]]))
    return catalog


def build_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    upstream, _ = make_upstream_tarball(tmp_path / "upstream")
    catalog = make_catalog(tmp_path / "catalog")
    return upstream, catalog


def build_two_identical_snapshots(tmp_path: Path) -> Tuple[dict, dict]:
    manifests = []
    for number in (1, 2):
        workdir = tmp_path / f"build{number}"
        upstream, catalog = build_fixture(workdir)
        manifests.append(
            build_snapshot(
                upstream_tarball=upstream,
                catalog_path=catalog,
                revision=1,
                output_dir=workdir / "out",
                manifest_path=workdir / "manifest.json",
            )
        )
    return manifests[0], manifests[1]


def test_build_snapshot_writes_pinned_manifest(tmp_path: Path) -> None:
    upstream, payloads = make_upstream_tarball(tmp_path / "upstream")
    catalog = make_catalog(tmp_path / "catalog")
    out = tmp_path / "out"

    manifest = build_snapshot(
        upstream_tarball=upstream,
        catalog_path=catalog,
        revision=1,
        output_dir=out,
        manifest_path=tmp_path / "evals/corpora/flores200-dev.json",
    )

    assert manifest["schema"] == "versta/corpus-manifest@1"
    assert manifest["revision"] == 1
    assert manifest["split"] == "dev"
    assert manifest["upstream"]["license"] == "CC-BY-SA-4.0"
    assert manifest["artifact"]["file"] == "flores200-dev-r1.tar.gz"
    assert manifest["artifact"]["urls"][0] == (
        "https://models.versta.app/evals/corpora/flores200-dev-r1.tar.gz"
    )
    assert set(manifest["language_map"].values()) == set(LANGUAGES.values())

    for code, data in payloads.items():
        entry = manifest["files"][f"dev/{code}.dev"]
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["sentences"] == 997

    directions = [(d["source"], d["target"]) for d in manifest["directions"]]
    assert directions == [("en", "es"), ("es", "en"), ("mt", "lv")]
    assert manifest["directions"][2]["base_model"] == "Firefox/mt-lv"

    artifact = out / "flores200-dev-r1.tar.gz"
    assert artifact.is_file()
    assert (out / "flores200-dev-r1.tar.sha256").read_text() == manifest["artifact"][
        "sha256"
    ]
    with tarfile.open(artifact, "r:gz") as tar:
        names = tar.getnames()
    assert "flores200-dev-r1/NOTICE" in names
    assert "flores200-dev-r1/manifest.json" in names
    assert "flores200-dev-r1/dev/eng_Latn.dev" in names

    # The written manifest passes its own fail-fast validation.
    load_manifest(tmp_path / "evals/corpora/flores200-dev.json")


def test_build_is_reproducible(tmp_path: Path) -> None:
    first, second = build_two_identical_snapshots(tmp_path)
    assert first["artifact"]["sha256"] == second["artifact"]["sha256"]
    assert first == second


def test_build_fails_on_missing_language(tmp_path: Path) -> None:
    languages = {k: v for k, v in LANGUAGES.items() if k != "mt"}
    upstream, _ = make_upstream_tarball(tmp_path / "upstream", languages)
    catalog = make_catalog(tmp_path / "catalog")

    with pytest.raises(ValueError, match="mlt_Latn"):
        build_snapshot(
            upstream, catalog, 1, tmp_path / "out", tmp_path / "manifest.json"
        )


def test_build_fails_on_wrong_sentence_count(tmp_path: Path) -> None:
    upstream, _ = make_upstream_tarball(tmp_path / "upstream", counts={"eng_Latn": 996})
    catalog = make_catalog(tmp_path / "catalog")

    with pytest.raises(ValueError, match="996 sentences"):
        build_snapshot(
            upstream, catalog, 1, tmp_path / "out", tmp_path / "manifest.json"
        )


def test_derive_directions_rejects_unmirrored_pair() -> None:
    entry = CatalogEntry(
        base_model="Firefox/en-es",
        source_language="en",
        target_language="es",
        bidirectional=True,
    )
    with pytest.raises(ValueError, match="not mirrored"):
        derive_directions([[entry]])


def serve_via_file_url(tmp_path: Path) -> Tuple[Path, Path]:
    """Builds a snapshot and returns (manifest_path, corpus_target_dir) wired
    to fetch the artifact over file:// — hermetic but real bytes."""
    upstream, catalog = build_fixture(tmp_path / "in")
    build_snapshot(
        upstream_tarball=upstream,
        catalog_path=catalog,
        revision=1,
        output_dir=tmp_path / "published",
        manifest_path=tmp_path / "manifest.json",
    )
    document = json.loads((tmp_path / "manifest.json").read_text())
    document["artifact"]["urls"] = [
        (tmp_path / "published" / document["artifact"]["file"]).as_uri()
    ]
    manifest_path = tmp_path / "fetch-manifest.json"
    manifest_path.write_text(json.dumps(document))
    return manifest_path, tmp_path / "corpus"


def test_fetch_reproduces_corpus_byte_identical(tmp_path: Path) -> None:
    manifest_path, _ = serve_via_file_url(tmp_path)

    first = fetch_snapshot(manifest_path, tmp_path / "machine-a", check=False)
    second = fetch_snapshot(manifest_path, tmp_path / "machine-b", check=False)

    # Byte-identical corpus out of the same manifest, on two "machines".
    first_files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert first_files == sorted(
        p.relative_to(second) for p in second.rglob("*") if p.is_file()
    )
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    # Re-running fetch on a populated cache re-verifies without downloading.
    again = fetch_snapshot(manifest_path, tmp_path / "machine-a", check=False)
    assert again == first


def test_fetch_check_flag(tmp_path: Path) -> None:
    manifest_path, _ = serve_via_file_url(tmp_path)

    with pytest.raises(ValueError, match="--check failed"):
        fetch_snapshot(manifest_path, tmp_path / "empty", check=True)

    corpus = fetch_snapshot(manifest_path, tmp_path / "empty", check=False)
    assert fetch_snapshot(manifest_path, tmp_path / "empty", check=True) == corpus


def test_fetch_rejects_tampered_artifact(tmp_path: Path) -> None:
    manifest_path, _ = serve_via_file_url(tmp_path)
    document = json.loads(manifest_path.read_text())
    artifact = Path(document["artifact"]["urls"][0][len("file://") :])
    blob = bytearray(artifact.read_bytes())
    blob[-1] ^= 0xFF
    artifact.write_bytes(bytes(blob))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch_snapshot(manifest_path, tmp_path / "corpus", check=False)
    assert not list((tmp_path / "corpus").rglob("*.dev"))


def test_verify_detects_tampered_file(tmp_path: Path) -> None:
    manifest_path, _ = serve_via_file_url(tmp_path)
    corpus = fetch_snapshot(manifest_path, tmp_path / "corpus", check=False)

    victim = corpus / "dev" / "eng_Latn.dev"
    victim.write_bytes(victim.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match=r"dev/eng_Latn\.dev: sha256 mismatch"):
        fetch_snapshot(manifest_path, tmp_path / "corpus", check=True)


def test_manifest_rejects_malformed_hash(tmp_path: Path) -> None:
    manifest_path, _ = serve_via_file_url(tmp_path)
    document = json.loads(manifest_path.read_text())
    document["artifact"]["sha256"] = "not-a-hash"
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="artifact.sha256"):
        load_manifest(manifest_path)


# --- Harness consumer side (versta.eval.corpora.slices) ---
#
# The producer tests above pin the snapshot lifecycle; these pin what the
# eval harness reads from it: verified per-direction sentence slices that
# fail closed on any drift between manifest and payload.


def _tamper_manifest(manifest_path: Path, mutate) -> Path:
    """Rewrites the fixture manifest with `mutate` applied to the document."""
    document = json.loads(manifest_path.read_text())
    mutate(document)
    manifest_path.write_text(json.dumps(document, indent=2) + "\n")
    return manifest_path


def test_harness_manifest_loads_checked_in_shape(
    eval_corpus: Tuple[Path, Path],
) -> None:
    manifest_path, _ = eval_corpus
    manifest = load_harness_manifest(manifest_path)

    assert manifest["name"] == "flores200-test"
    assert manifest["split"] == "dev"
    assert manifest["revision"] == 1
    assert (
        snapshot_dir(manifest, eval_corpus[1]) == eval_corpus[1] / "flores200-test-r1"
    )


def test_harness_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="not found"):
        load_harness_manifest(tmp_path / "nope.json")


def test_harness_manifest_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_harness_manifest(path)


def test_harness_manifest_rejects_wrong_schema(eval_corpus: Tuple[Path, Path]) -> None:
    manifest_path, _ = eval_corpus
    _tamper_manifest(manifest_path, lambda doc: doc.update(schema="bogus/9"))

    with pytest.raises(CorpusError, match="versta/corpus-manifest@1"):
        load_harness_manifest(manifest_path)


def test_harness_manifest_rejects_bad_digest(eval_corpus: Tuple[Path, Path]) -> None:
    manifest_path, _ = eval_corpus

    def mutate(doc: dict) -> None:
        doc["files"]["dev/eng_Latn.dev"]["sha256"] = "not-a-sha"

    _tamper_manifest(manifest_path, mutate)
    with pytest.raises(CorpusError, match="sha256"):
        load_harness_manifest(manifest_path)


def test_resolve_slice_reads_verified_sentences(eval_corpus: Tuple[Path, Path]) -> None:
    manifest_path, corpus_root = eval_corpus
    manifest = load_harness_manifest(manifest_path)
    corpus = resolve_slice(manifest, "en-es", snapshot_dir(manifest, corpus_root))

    assert corpus.direction == "en-es"
    assert corpus.corpus == "flores200-test"
    assert corpus.split == "dev"
    assert corpus.revision == "r1"
    assert corpus.flores_source == "eng_Latn"
    assert corpus.flores_target == "spa_Latn"
    assert corpus.source_sentences == [pair[0] for pair in EVAL_PAIRS]
    assert corpus.target_sentences == [pair[1] for pair in EVAL_PAIRS]
    assert len(corpus.source_sha256) == 64
    assert len(corpus.target_sha256) == 64


def test_resolve_slice_unknown_direction_lists_available(
    eval_corpus: Tuple[Path, Path],
) -> None:
    manifest_path, corpus_root = eval_corpus
    manifest = load_harness_manifest(manifest_path)

    with pytest.raises(CorpusError, match="available: en-es"):
        resolve_slice(manifest, "en-fr", snapshot_dir(manifest, corpus_root))


def test_resolve_slice_fails_on_payload_drift(eval_corpus: Tuple[Path, Path]) -> None:
    manifest_path, corpus_root = eval_corpus
    manifest = load_harness_manifest(manifest_path)
    snapshot = snapshot_dir(manifest, corpus_root)

    victim = snapshot / "dev" / "eng_Latn.dev"
    victim.write_bytes(victim.read_bytes().replace(b"museum", b"museuM", 1))

    with pytest.raises(CorpusError, match="hash mismatch"):
        resolve_slice(manifest, "en-es", snapshot)


def test_resolve_slice_fails_on_missing_payload(eval_corpus: Tuple[Path, Path]) -> None:
    manifest_path, corpus_root = eval_corpus
    manifest = load_harness_manifest(manifest_path)
    snapshot = snapshot_dir(manifest, corpus_root)
    (snapshot / "dev" / "spa_Latn.dev").unlink()

    with pytest.raises(CorpusError, match="fetch the pinned corpus"):
        resolve_slice(manifest, "en-es", snapshot)


def test_resolve_slice_rejects_blank_source_lines(tmp_path: Path) -> None:
    pairs = [("Hello.", "Hola."), ("", "Vacia.")]
    manifest_path, corpus_root = write_eval_corpus(tmp_path, pairs=pairs)
    manifest = load_harness_manifest(manifest_path)

    with pytest.raises(CorpusError, match="blank lines"):
        resolve_slice(manifest, "en-es", snapshot_dir(manifest, corpus_root))


def test_resolve_slice_rejects_sentence_count_drift(tmp_path: Path) -> None:
    manifest_path, corpus_root = write_eval_corpus(tmp_path)

    def mutate(doc: dict) -> None:
        # Keep the manifest internally consistent (producer validation
        # cross-checks direction vs. file counts); the drift under test is
        # manifest vs. payload.
        doc["directions"][0]["sentences"] = len(EVAL_PAIRS) + 1
        doc["files"]["dev/eng_Latn.dev"]["sentences"] = len(EVAL_PAIRS) + 1
        doc["files"]["dev/spa_Latn.dev"]["sentences"] = len(EVAL_PAIRS) + 1

    _tamper_manifest(manifest_path, mutate)
    manifest = load_harness_manifest(manifest_path)
    with pytest.raises(CorpusError, match="sentences"):
        resolve_slice(manifest, "en-es", snapshot_dir(manifest, corpus_root))
