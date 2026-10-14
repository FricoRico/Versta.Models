import gzip
import json
import struct

from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from versta.leanmt import LEANMT_BINARY_VERSION


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
) -> Path:
    """
    Writes a complete, schema-valid download output directory for `direction`
    (e.g. "en-es") as versta.download would have produced it. Files listed in
    `missing` are referenced by metadata but omitted from disk. `tensor_names`
    steers the fake model's tensor table (two-vocabulary by default).
    """
    source, target = direction.split("-")
    folder = base / direction
    folder.mkdir(parents=True)

    files = {
        "model": "model.bin",
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
