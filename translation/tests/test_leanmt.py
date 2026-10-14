import io
import struct
from pathlib import Path

import pytest

from conftest import (
    SHARED_VOCAB_TENSORS,
    SPLIT_VOCAB_TENSORS,
    bergamot_model_bin,
)
from versta.leanmt import (
    LEANMT_BINARY_VERSION,
    LeanmtFormatError,
    assert_leanmt_compatible,
    model_requires_target_vocabulary,
    parse_binary_model_version,
    read_binary_model_version,
    read_tensor_names,
    read_tensor_table,
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


class TestReadTensorTable:
    def test_parses_version_and_names(self) -> None:
        stream = io.BytesIO(bergamot_model_bin(label=b"rest is data"))
        version, names = read_tensor_table(stream, "test")
        assert version == LEANMT_BINARY_VERSION
        assert names == list(SPLIT_VOCAB_TENSORS)

    def test_reads_names_from_disk(self, tmp_path: Path) -> None:
        model = tmp_path / "model.bin"
        model.write_bytes(bergamot_model_bin(SHARED_VOCAB_TENSORS))
        assert read_tensor_names(model) == list(SHARED_VOCAB_TENSORS)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            read_tensor_names(tmp_path / "model.bin")

    def test_fails_closed_on_foreign_version(self) -> None:
        """A reader written for this layout must not guess at foreign ones."""
        stream = io.BytesIO(bergamot_model_bin(version=2))
        with pytest.raises(LeanmtFormatError, match="binary model version mismatch"):
            read_tensor_table(stream, "test")

    def test_rejects_implausible_header_count(self) -> None:
        blob = struct.pack("<Q", 1) + struct.pack("<Q", 10**9) + b"\x00" * 64
        with pytest.raises(LeanmtFormatError, match="sanity bound"):
            read_tensor_table(io.BytesIO(blob), "test")

    @pytest.mark.parametrize("cut", [100, 124])
    def test_rejects_truncated_table(self, cut: int) -> None:
        # bergamot_model_bin: 8B version + 8B count + 3x32B headers (=112),
        # then 13B first name — cut=100 lands mid-headers, cut=124 mid-names.
        blob = bergamot_model_bin()[:cut]
        with pytest.raises(LeanmtFormatError, match="corrupt or truncated"):
            read_tensor_table(io.BytesIO(blob), "test")

    def test_rejects_non_nul_terminated_name(self) -> None:
        blob = bergamot_model_bin().replace(b"encoder_Wemb\x00", b"encoder_WembX")
        with pytest.raises(LeanmtFormatError, match="not NUL-terminated"):
            read_tensor_table(io.BytesIO(blob), "test")


class TestModelRequiresTargetVocabulary:
    def test_split_wemb_models_require_it(self) -> None:
        assert model_requires_target_vocabulary(SPLIT_VOCAB_TENSORS) is True
        assert model_requires_target_vocabulary(("decoder_Wemb",)) is True

    def test_shared_wemb_models_do_not(self) -> None:
        assert model_requires_target_vocabulary(SHARED_VOCAB_TENSORS) is False

    def test_quantmult_scalar_alone_does_not(self) -> None:
        # The auxiliary QuantMultA scalars only accompany a base embedding
        # tensor; on their own they are not evidence of a split vocabulary.
        assert model_requires_target_vocabulary(("decoder_Wemb_QuantMultA",)) is False
