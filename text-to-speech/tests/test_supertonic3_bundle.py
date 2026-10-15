"""Bundle assembly, lint and reproducibility (NEU-68 / OpenRAIL obligations)."""

import json
import os
import tarfile

from pathlib import Path

import pytest

from versta.supertonic3 import build_bundle
from versta.supertonic3.build_bundle import bundle, lint_bundle, write_tarball
from versta.supertonic3.config import REQUIRED_BUNDLE_FILES
from versta.supertonic3.typing import MODEL_STAGES

STYLE_NAMES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


def _fake_converted(tmp_path: Path) -> Path:
    """A minimal but complete conversion output tree."""
    converted = tmp_path / "converted"
    for stage in MODEL_STAGES:
        stage_dir = converted / "int8"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / f"{stage}.mnn").write_bytes(b"mnn-weights-" + stage.encode())
    for relative in ("tts.json", "unicode_indexer.json", "LICENSE"):
        target = converted / "passthrough" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(("content-of-" + relative).encode())
    for name in STYLE_NAMES:
        target = converted / "passthrough" / "voice_styles" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{}")
    record = {
        "converted_at": "2026-10-15T00:00:00+00:00",
        "upstream": {"repo": "Supertone/supertonic-3", "revision": "deadbeef"},
        "toolchain": {"mnn_commit": "d407447e", "mnn_tag": "3.6.1", "python": "3.12"},
        "artifacts": [],
    }
    (converted / "conversion-record.json").write_text(json.dumps(record))
    return converted


def test_full_bundle_contains_every_required_file(tmp_path):
    tarball, checksum = bundle(_fake_converted(tmp_path), tmp_path / "out", "v2.0.0")
    assert checksum.exists()
    with tarfile.open(tarball) as tar:
        names = sorted(tar.getnames())
    assert names == sorted(REQUIRED_BUNDLE_FILES)
    with tarfile.open(tarball) as tar:
        readme = tar.extractfile("README.md").read().decode()
        metadata = json.loads(tar.extractfile("metadata.json").read().decode())
    assert metadata["version"] == "v2.0.0"
    assert "Open RAIL-M" in readme and "deadbeef" in readme


def test_bundle_is_byte_reproducible(tmp_path):
    converted = _fake_converted(tmp_path)
    first, _ = bundle(converted, tmp_path / "a", "v2.0.0")
    second, _ = bundle(converted, tmp_path / "b", "v2.0.0")
    assert first.read_bytes() == second.read_bytes()


def test_lint_fails_closed_on_missing_license(tmp_path):
    staging = tmp_path / "supertonic3"
    build_bundle.assemble_bundle_dir(_fake_converted(tmp_path), staging)
    (staging / "LICENSE").unlink()
    assert any("LICENSE" in v for v in lint_bundle(staging))


def test_assemble_fails_fast_on_missing_stage(tmp_path):
    converted = _fake_converted(tmp_path)
    (converted / "int8" / "vocoder.mnn").unlink()
    with pytest.raises(FileNotFoundError):
        build_bundle.assemble_bundle_dir(converted, tmp_path / "staging")


def test_tarball_enforces_size_cap(tmp_path, monkeypatch):
    staging = tmp_path / "st"
    staging.mkdir()
    # Incompressible bytes so gzip cannot shrink the tarball under the cap.
    (staging / "big.bin").write_bytes(os.urandom(4096))
    monkeypatch.setattr(build_bundle, "BUNDLE_SIZE_CAP_BYTES", 1024)
    with pytest.raises(RuntimeError, match="120 MB int8 cap"):
        write_tarball(staging, tmp_path / "out")
