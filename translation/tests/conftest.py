import gzip
import json
import socket
import struct

from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from versta.leanmt import LEANMT_BINARY_VERSION
from versta.eval.typing import EvalResult


# --- Hermetic suite guard: no real network on the PR path (NEU-53) ---------


class NetworkBlockedError(RuntimeError):
    """A test attempted a real network connection; the suite is hermetic."""


def _blocked_network(*args: object, **kwargs: object) -> None:
    raise NetworkBlockedError(
        "test attempted a real network connection. The PR-CI suite is "
        "hermetic by construction (mocked registry, tiny synthetic fixture "
        "corpus, fake engine) — mock the network boundary in the test "
        "instead. Real-network model evaluation runs only in the "
        "release-candidate workflow "
        "(.github/workflows/release-eval.yml)."
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fails any outbound connection attempt from a test, fast and
    deterministically. The whole suite is designed hermetic — fixtures mock
    the download boundary (urlopen), the corpus fetch and the engine — so a
    real connection means a boundary regression that must not flake CI on
    network state (or, worse, silently evaluate against live upstream data).
    """
    monkeypatch.setattr(socket, "create_connection", _blocked_network)
    monkeypatch.setattr(socket.socket, "connect", _blocked_network)


class FakeResponse:
    """Minimal stand-in for http.client.HTTPResponse as used by urlopen()."""

    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


# Remote file name -> registry file kind.
FILE_KINDS = {
    "model.bin": "model",
    "lex.bin": "lexicalShortlist",
    "vocab.spm": "srcVocab",
    "vocab.trg.spm": "trgVocab",
}

# Tensor tables the fixtures can model, mirroring the real bergamot exports:
# two-vocabulary models (e.g. en-ja) carry split encoder/decoder embeddings,
# shared-vocabulary models (e.g. en-es) carry a single Wemb. The auxiliary
# QuantMultA scalars accompany their embeddings in the real files.
SPLIT_VOCAB_TENSORS = ("encoder_Wemb", "decoder_Wemb", "decoder_Wemb_QuantMultA")
SHARED_VOCAB_TENSORS = ("Wemb", "Wemb_QuantMultA")

# Wire value of marian's legacy int8-quantized tensor type as written by its
# exporter; the leanmt loader intercepts it (src/Io.cc in Versta.Leanmt).
TENSOR_TYPE_INTGEMM8 = 0x4101

_TENSOR_DIM = 64


def bergamot_model_bin(
    tensor_names: Tuple[str, ...] = SPLIT_VOCAB_TENSORS,
    version: int = LEANMT_BINARY_VERSION,
    label: bytes = b"",
) -> bytes:
    """
    Builds a structurally valid bergamot/leanmt binary model skeleton: the
    uint64 format version, the tensor table (headers, NUL-terminated names,
    shapes), the data-section offset and a zeroed data section — the layout
    ``load_items`` in Versta.Leanmt src/Io.cc walks. `label` is appended after
    the data section so payloads stay distinguishable per fixture direction.
    """
    blob = struct.pack("<Q", version)
    blob += struct.pack("<Q", len(tensor_names))
    for name in tensor_names:
        blob += struct.pack(
            "<QQQQ", len(name) + 1, TENSOR_TYPE_INTGEMM8, 1, _TENSOR_DIM
        )
    for name in tensor_names:
        blob += name.encode("utf-8") + b"\x00"
    for _ in tensor_names:
        blob += struct.pack("<i", _TENSOR_DIM)
    blob += struct.pack("<Q", 0)
    blob += b"\x00" * (_TENSOR_DIM * len(tensor_names))
    return blob + label


def fake_model_payload(
    kind: str,
    source: str,
    target: str,
    tensor_names: Tuple[str, ...] = SPLIT_VOCAB_TENSORS,
) -> bytes:
    payload = f"payload:{source}-{target}:{kind}".encode("utf-8")
    if kind == "model":
        # Bergamot binary models open with the uint64 format version and the
        # tensor table the leanmt loader walks; keep fixtures structurally
        # realistic so the pipeline's fail-closed schema gates can parse them.
        return bergamot_model_bin(tensor_names, label=payload)
    return payload


def make_registry_entry(source: str, target: str, architecture: str = "tiny") -> dict:
    """Builds a Mozilla-registry entry with deterministic fake payloads."""
    prefix = f"models/{source}-{target}/{architecture}"
    files: Dict[str, dict] = {}

    for name, kind in FILE_KINDS.items():
        payload = fake_model_payload(kind, source, target)
        files[kind] = {
            "path": f"{prefix}/{name}.gz",
            "uncompressedHash": sha256(payload).hexdigest(),
        }

    return {
        "architecture": architecture,
        "sourceLanguage": source,
        "targetLanguage": target,
        "files": files,
        "metrics": {"flores200-plus": {"comet22": 0.874}},
    }


def make_registry(pairs: List[Tuple[str, str]]) -> dict:
    """Builds a full registry document covering every requested direction."""
    return {
        "baseUrl": "https://models.example/",
        "models": {
            f"{source}-{target}": [make_registry_entry(source, target)]
            for source, target in pairs
        },
    }


REGISTRY_URL = "https://models.example/registry.json"


def model_metadata_payload(source: str, target: str) -> bytes:
    """Per-model metadata.json as published next to the model file upstream."""
    return json.dumps(
        {
            "modelConfig": {
                "enc-depth": 6,
                # Mirrors the real registry: the current en-ja/ja-en exports
                # (remote-settings model v2.3 generation) carry dec-depth 4,
                # most other pairs carry 2.
                "dec-depth": 4 if "ja" in (source, target) else 2,
                "transformer-ffn-depth": 2,
                "transformer-heads": 8,
                "version": "v1.12.14 2d067af 2024-02-16 11:44:13 -0500",
            }
        }
    ).encode("utf-8")


@pytest.fixture()
def mock_registry(monkeypatch: pytest.MonkeyPatch) -> Dict[str, dict]:
    """
    Routes every urlopen() call in versta.download.download to deterministic
    fake payloads: the registry document, gzipped model files and per-model
    metadata. Covers the en-es and es-en directions by default. Fully offline.
    """
    registry = make_registry([("en", "es"), ("es", "en")])
    registry_payload = json.dumps(registry).encode("utf-8")

    def fake_urlopen(request, timeout: int = 0) -> FakeResponse:
        url = request.full_url

        if url == REGISTRY_URL:
            return FakeResponse(registry_payload)

        if url.endswith("/metadata.json"):
            direction = url.split("/models/")[1].split("/")[0]
            source, target = direction.split("-")
            return FakeResponse(model_metadata_payload(source, target))

        if "/models/" in url and url.endswith(".gz"):
            relative = url.split("/models/")[1]
            direction, _architecture, compressed_name = relative.split("/")
            source, target = direction.split("-")
            name = compressed_name[: -len(".gz")]
            kind = FILE_KINDS[name]
            return FakeResponse(gzip.compress(fake_model_payload(kind, source, target)))

        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("versta.download.download.urlopen", fake_urlopen)
    return registry


def write_direction_dir(
    base: Path,
    direction: str,
    missing: Tuple[str, ...] = (),
    tensor_names: Tuple[str, ...] = SPLIT_VOCAB_TENSORS,
    model_name: str = "model.bin",
) -> Path:
    """
    Writes a complete, schema-valid download output directory for `direction`
    (e.g. "en-es") as versta.download would have produced it. Files listed in
    `missing` are referenced by metadata but omitted from disk. `tensor_names`
    steers the fake model's tensor table (two-vocabulary by default).
    `model_name` defaults to the historical short name; eval fixtures pass a
    realistic upstream name so the harness's gemm-precision resolution
    (keyed on the quantized-model suffix) applies.
    """
    source, target = direction.split("-")
    folder = base / direction
    folder.mkdir(parents=True)

    files = {
        "model": model_name,
        "vocabulary": "vocab.spm",
        "target_vocabulary": "vocab.trg.spm",
        "shortlist": "lex.bin",
    }
    for kind, name in files.items():
        if kind in missing:
            continue
        (folder / name).write_bytes(
            fake_model_payload(kind, source, target, tensor_names)
        )

    metadata = {
        "directory": direction,
        "source_language": source,
        "target_language": target,
        "architecture": "tiny",
        "base_model": f"{direction}:tiny",
        "score": 0.874,
        "version": "v1.12.14",
        "files": files,
        "config": {
            "encoder_layers": 6,
            # Mirrors the real registry: ja directions of the current
            # generation carry dec-depth 4.
            "decoder_layers": 4 if "ja" in (source, target) else 2,
            "ffn_depth": 2,
            "num_heads": 8,
            "split_mode": "sentence",
        },
    }
    with open(folder / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    return folder


# --- versta.eval fixtures: pinned-corpus manifest, fake engine, bundles ---

# A tiny stand-in for the pinned FLORES-200 snapshot: a handful of sentence
# pairs (including digits, so output-affecting defects are observable). The
# fake engine's sentence-level dictionary reproduces the target side
# exactly, so a correct engine scores BLEU/chrF 100 and a mangling engine
# scores measurably lower.
EVAL_PAIRS: List[Tuple[str, str]] = [
    ("The museum opens at 9 in the morning.", "El museo abre a las 9 de la mañana."),
    ("I need 2 tickets for the train.", "Necesito 2 billetes para el tren."),
    ("Where is the nearest pharmacy?", "¿Dónde está la farmacia más cercana?"),
    (
        "The hotel has 42 rooms on 5 floors.",
        "El hotel tiene 42 habitaciones en 5 plantas.",
    ),
    ("Lunch is served at half past 12.", "El almuerzo se sirve a las 12 y media."),
    ("Can you help me please?", "¿Puede ayudarme por favor?"),
]


# Tiny catalog code -> FLORES-200 code map mirroring the real snapshot's;
# the fixture directions only ever cover English and Spanish.
EVAL_FLORES_CODES = {"en": "eng_Latn", "es": "spa_Latn"}


def write_eval_corpus(
    base: Path,
    direction: str = "en-es",
    pairs: List[Tuple[str, str]] = EVAL_PAIRS,
    manifest_overrides: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    """
    Writes a tiny pinned-corpus tree under `base` in the exact layout the
    corpus pinning task produces (NEU-49): payload files under a snapshot
    directory plus a versta/corpus-manifest@1 manifest with real SHA-256
    hashes. Returns (manifest path, corpus fetch-cache root) — pass them to
    the harness as corpus_manifest_path / corpus_root.
    """
    source, target = direction.split("-")
    flores_source = EVAL_FLORES_CODES[source]
    flores_target = EVAL_FLORES_CODES[target]

    corpus_root = base / "corpora"
    snapshot = corpus_root / "flores200-test-r1"
    snapshot.mkdir(parents=True)

    def write_payload(code: str, lines: List[str]) -> Dict[str, object]:
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        (snapshot / "dev").mkdir(exist_ok=True)
        (snapshot / "dev" / f"{code}.dev").write_bytes(payload)
        return {"sha256": sha256(payload).hexdigest(), "sentences": len(lines)}

    source_rel = f"dev/{flores_source}.dev"
    target_rel = f"dev/{flores_target}.dev"
    files: Dict[str, Dict[str, object]] = {}
    files[source_rel] = write_payload(flores_source, [pair[0] for pair in pairs])
    files.setdefault(
        target_rel, write_payload(flores_target, [pair[1] for pair in pairs])
    )

    manifest: Dict[str, object] = {
        "schema": "versta/corpus-manifest@1",
        "name": "flores200-test",
        "revision": 1,
        "split": "dev",
        "upstream": {
            "name": "FLORES-200",
            "url": "https://tests.invalid/flores200_dataset.tar.gz",
            "sha256": "0" * 64,
            "size_bytes": 0,
            "license": "CC-BY-SA-4.0",
            "attribution": "fixture attribution",
        },
        "artifact": {
            "file": "flores200-test-r1.tar.gz",
            "sha256": "0" * 64,
            "size_bytes": 0,
            "urls": ["https://tests.invalid/flores200-test-r1.tar.gz"],
        },
        "language_map": {"en": "eng_Latn", "es": "spa_Latn"},
        "files": files,
        "directions": [
            {
                "source": source,
                "target": target,
                "base_model": f"Firefox/{direction}",
                "source_file": source_rel,
                "target_file": target_rel,
                "sentences": len(pairs),
            }
        ],
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)

    manifest_path = base / "evals" / "corpora" / "flores200-test.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, corpus_root


_FAKE_BERGAMOT_TEMPLATE = '''#!/usr/bin/env python3
"""Fake bergamot CLI for hermetic eval tests: honors the CLI contract
(--model-config-paths, --log-level, --version; sentences on stdin,
line-aligned translations on stdout) with a deterministic embedded
sentence dictionary instead of a real model."""
import json
import re
import sys

MAPPING = json.loads({mapping_json})
MODE = {mode_literal}


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("bergamot-fake 0.0.0-test")
        return 0
    if MODE == "exit":
        sys.stderr.write("fake engine: forced failure\\n")
        return 3
    lines = sys.stdin.read().splitlines()
    if MODE == "drop-last":
        lines = lines[:-1]
    output = []
    for line in lines:
        translated = MAPPING.get(line, line)
        if MODE == "digit-mangle":
            translated = re.sub(r"[0-9]", "#", translated)
        output.append(translated)
    sys.stdout.write("\\n".join(output) + "\\n")
    return 0


sys.exit(main())
'''


def write_fake_bergamot(path: Path, mode: str = "faithful") -> Path:
    """
    Writes an executable fake bergamot CLI at `path`. Modes: "faithful"
    (reproduces the EVAL_PAIRS Spanish side exactly), "digit-mangle" (the
    NEU-30 class: corrupts digits in translations), "drop-last" (violates
    line alignment), "exit" (fails with a nonzero exit code).
    """
    mapping = {source: target for source, target in EVAL_PAIRS}
    script = _FAKE_BERGAMOT_TEMPLATE.format(
        mapping_json=repr(json.dumps(mapping, ensure_ascii=False)),
        mode_literal=repr(mode),
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_eval_bundle(base: Path, direction: str = "en-es") -> Path:
    """
    Packs a schema-valid single-direction bundle tarball the way
    versta.bundle does (direction directory plus bundle-level
    metadata.json at the archive root) and returns its path.
    """
    from versta.bundle.bundle_tar import bundle_files
    from versta.leanmt import LEANMT_BINARY_VERSION

    source, target = direction.split("-")
    staging = base / "staging"
    write_direction_dir(
        staging, direction, model_name=f"model.{source}{target}.intgemm.alphas.bin"
    )

    bundle_metadata = {
        "version": "v2.1.0",
        "leanmt_binary_version": LEANMT_BINARY_VERSION,
        "languages": sorted({source, target}),
        "bidirectional": False,
        "metadata": [
            {
                "directory": direction,
                "source_language": source,
                "target_language": target,
            }
        ],
    }
    (staging / "metadata.json").write_text(
        json.dumps(bundle_metadata, indent=4), encoding="utf-8"
    )

    return bundle_files(sorted(staging.iterdir()), base / f"{direction}-bundle.tar.gz")


@pytest.fixture()
def eval_corpus(tmp_path: Path) -> Tuple[Path, Path]:
    """A verified tiny corpus: (manifest path, fetch-cache root) for en-es."""
    return write_eval_corpus(tmp_path)


@pytest.fixture()
def fake_bergamot(tmp_path: Path) -> Path:
    """An executable fake bergamot CLI translating the fixture corpus."""
    return write_fake_bergamot(tmp_path / "bergamot")


# --- versta.eval.gate fixtures: synthetic result/ledger documents ---


def make_eval_result(
    direction: str = "en-es",
    version: str = "v2.0.0",
    bleu: float = 41.25,
    chrf: float = 60.1,
    comet22: Optional[float] = None,
    corpus_revision: str = "tests/fixture@0001",
    engine_version: str = "bergamot-fake 0.0.0-test",
    size_bytes: int = 24_000_000,
    started_utc: str = "2026-10-13T12:00:00Z",
) -> EvalResult:
    """
    Builds a schema-valid versta.eval-result/1 document with deterministic
    hashes, so gate/ledger tests need no engine, corpus payloads or network.
    Defaults model the en-es tiny direction at the shipped v2.0.0.
    """
    source, target = direction.split("-")
    digest = sha256(f"{direction}:{version}".encode("utf-8")).hexdigest()
    model = {
        "base_model": f"{direction}:tiny",
        "version": version,
        "architecture": "tiny",
        "model_sha256": digest,
        "metadata_sha256": sha256(f"metadata:{digest}".encode()).hexdigest(),
        "bundle_file": f"{direction}-bundle.tar.gz",
        "bundle_sha256": sha256(f"bundle:{digest}".encode()).hexdigest(),
        "size_bytes": size_bytes,
    }
    return EvalResult(
        schema="versta.eval-result/1",
        direction=direction,
        source_language=source,
        target_language=target,
        model=model,
        corpus={
            "corpus": "flores-200",
            "split": "dev",
            "revision": corpus_revision,
            "sentences": len(EVAL_PAIRS),
            "source_sha256": sha256(f"src:{corpus_revision}".encode()).hexdigest(),
            "target_sha256": sha256(f"tgt:{corpus_revision}".encode()).hexdigest(),
        },
        engine={
            "name": "bergamot-translator",
            "variant": "local",
            "version": engine_version,
            "decode": {"beam_size": 1},
            "config_sha256": sha256(b"decode-config").hexdigest(),
        },
        metrics={
            "sacrebleu_version": "2.5.1",
            "bleu_signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1",
            "chrf_signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1",
            "comet22_model": "Unbabel/wmt22-comet-da" if comet22 is not None else None,
        },
        scores={"bleu": bleu, "chrf": chrf, "comet22": comet22},
        run={
            "started_utc": started_utc,
            "duration_seconds": 1.5,
            "hypothesis_sha256": sha256(f"hyp:{digest}".encode()).hexdigest(),
        },
    )
