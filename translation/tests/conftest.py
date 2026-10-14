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


def fake_model_payload(kind: str, source: str, target: str) -> bytes:
    payload = f"payload:{source}-{target}:{kind}".encode("utf-8")
    if kind == "model":
        # Bergamot binary models open with the uint64 format version the leanmt
        # loader checks (kBinaryFileVersion); keep the fixture header realistic.
        return struct.pack("<Q", LEANMT_BINARY_VERSION) + payload
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
                "dec-depth": 2,
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
    base: Path, direction: str, missing: Tuple[str, ...] = ()
) -> Path:
    """
    Writes a complete, schema-valid download output directory for `direction`
    (e.g. "en-es") as versta.download would have produced it. Files listed in
    `missing` are referenced by metadata but omitted from disk.
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
        (folder / name).write_bytes(fake_model_payload(kind, source, target))

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
            "decoder_layers": 2,
            "ffn_depth": 2,
            "num_heads": 8,
            "split_mode": "sentence",
        },
    }
    with open(folder / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    return folder
