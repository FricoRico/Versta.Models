import struct
from pathlib import Path

import pytest

from versta.leanmt import (
    LEANMT_BINARY_VERSION,
    LeanmtFormatError,
    assert_leanmt_compatible,
    parse_binary_model_version,
    read_binary_model_version,
)


def _write_model(path: Path, version: int = LEANMT_BINARY_VERSION) -> Path:
    """Writes a minimal bergamot-style model file: version header + payload."""
    path.write_bytes(struct.pack("<Q", version) + b"tensor-bytes")
    return path


class TestParseBinaryModelVersion:
    def test_reads_little_endian_uint64(self) -> None:
        assert parse_binary_model_version(struct.pack("<Q", 1) + b"x", "test") == 1

    def test_rejects_truncated_header(self) -> None:
        with pytest.raises(LeanmtFormatError, match="truncated model file"):
            parse_binary_model_version(b"\x01\x00\x00", "test")

    def test_rejects_empty_header(self) -> None:
        with pytest.raises(LeanmtFormatError, match="truncated model file"):
            parse_binary_model_version(b"", "test")


class TestReadBinaryModelVersion:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            read_binary_model_version(tmp_path / "model.bin")

    def test_reads_version_from_disk(self, tmp_path: Path) -> None:
        model = _write_model(tmp_path / "model.bin", version=7)
        assert read_binary_model_version(model) == 7


class TestAssertLeanmtCompatible:
    def test_accepts_current_version(self, tmp_path: Path) -> None:
        assert_leanmt_compatible(_write_model(tmp_path / "model.bin"))

    def test_fails_closed_on_other_versions(self, tmp_path: Path) -> None:
        model = _write_model(tmp_path / "model.bin", version=2)
        with pytest.raises(LeanmtFormatError, match=r"embedded v2, leanmt expects v1"):
            assert_leanmt_compatible(model)

    def test_fails_closed_on_truncated_file(self, tmp_path: Path) -> None:
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x01\x00")
        with pytest.raises(LeanmtFormatError, match="truncated model file"):
            assert_leanmt_compatible(model)


def test_version_constant_mirrors_leanmt_loader() -> None:
    """LEANMT_BINARY_VERSION tracks kBinaryFileVersion in Versta.Leanmt
    (src/Io.hh); bumping it is a negotiated format change (NEU-25)."""
    assert LEANMT_BINARY_VERSION == 1
