from pathlib import Path

import pytest

from versta.eval.config import (
    DEFAULT_DECODE_CONFIG,
    EvalConfigError,
    resolve_gemm_precision,
)
from versta.eval.engines import (
    BergamotEngine,
    EngineError,
    _render_config,
    create_engine,
)
from versta.eval.models import DirectionInput, load_direction_dir

from conftest import EVAL_PAIRS, write_direction_dir, write_fake_bergamot


def _direction(base: Path) -> DirectionInput:
    return load_direction_dir(
        write_direction_dir(
            base / "model", "en-es", model_name="model.enes.intgemm.alphas.bin"
        )
    )


def test_resolve_gemm_precision_known_names() -> None:
    assert (
        resolve_gemm_precision("model.enes.intgemm.alphas.bin") == "int8shiftAlphaAll"
    )
    assert resolve_gemm_precision("model.deden.intgemm8.bin") == "int8shiftAll"


def test_resolve_gemm_precision_fails_closed_on_unknown_name() -> None:
    with pytest.raises(EvalConfigError, match="gemm-precision"):
        resolve_gemm_precision("model.bin")


def test_render_config_matches_pinned_production_settings() -> None:
    expected = (
        "bergamot-mode: wasm\n"
        "models:\n"
        "  - model/model.bin\n"
        "vocabs:\n"
        "  - model/vocab.src.spm\n"
        "  - model/vocab.trg.spm\n"
        "shortlist:\n"
        "  - model/lex.bin\n"
        "  - false\n"
        "beam-size: 1\n"
        "normalize: 1.0\n"
        "word-penalty: 0.0\n"
        "max-length-break: 128\n"
        "mini-batch-words: 1024\n"
        "workspace: 128\n"
        "max-length-factor: 2.0\n"
        "skip-cost: true\n"
        "cpu-threads: 4\n"
        "quiet: false\n"
        "quiet-translation: false\n"
        "gemm-precision: int8shiftAlphaAll\n"
        "alignment: soft\n"
        "ssplit-mode: sentence\n"
        "max-input-sentence-tokens: 1024\n"
        "max-input-tokens: 2048000\n"
    )
    assert _render_config(DEFAULT_DECODE_CONFIG, "int8shiftAlphaAll") == expected


def test_engine_requires_exactly_one_runtime(tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="exactly one bergamot runtime"):
        BergamotEngine()
    with pytest.raises(EngineError, match="exactly one bergamot runtime"):
        BergamotEngine(
            binary=tmp_path / "bin", image="registry.example/bergamot:latest"
        )


def test_engine_fails_on_missing_binary(tmp_path: Path) -> None:
    engine = BergamotEngine(binary=tmp_path / "missing-bergamot")
    assert engine.version() == "unknown"
    with pytest.raises(EngineError, match="binary not found"):
        engine.translate(
            _direction(tmp_path),
            [pair[0] for pair in EVAL_PAIRS],
            tmp_path / "work",
        )


def test_engine_translates_line_aligned(tmp_path: Path, fake_bergamot: Path) -> None:
    engine = BergamotEngine(binary=fake_bergamot)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    outcome = engine.translate(
        _direction(tmp_path), [pair[0] for pair in EVAL_PAIRS], work_dir
    )
    assert outcome.translations == [pair[1] for pair in EVAL_PAIRS]
    assert outcome.variant == "local"
    assert outcome.version == "bergamot-fake 0.0.0-test"
    assert len(outcome.config_sha256) == 64
    assert outcome.decode["beam-size"] == 1
    assert outcome.decode["ssplit-mode"] == "sentence"


def test_engine_config_is_stable_across_runs(
    tmp_path: Path, fake_bergamot: Path
) -> None:
    engine = BergamotEngine(binary=fake_bergamot)
    hashes = []
    for run in ("run-a", "run-b"):
        work_dir = tmp_path / run
        work_dir.mkdir()
        outcome = engine.translate(
            _direction(tmp_path / "inputs" / run),
            [pair[0] for pair in EVAL_PAIRS],
            work_dir,
        )
        hashes.append(outcome.config_sha256)
    assert hashes[0] == hashes[1]


def test_engine_rejects_misaligned_output(tmp_path: Path) -> None:
    engine = BergamotEngine(
        binary=write_fake_bergamot(tmp_path / "bergamot", mode="drop-last")
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(EngineError, match="not line-aligned"):
        engine.translate(
            _direction(tmp_path), [pair[0] for pair in EVAL_PAIRS], work_dir
        )


def test_engine_propagates_failure_with_stderr(tmp_path: Path) -> None:
    engine = BergamotEngine(
        binary=write_fake_bergamot(tmp_path / "bergamot", mode="exit")
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(EngineError, match="forced failure"):
        engine.translate(
            _direction(tmp_path), [pair[0] for pair in EVAL_PAIRS], work_dir
        )


def test_create_engine_unknown_name() -> None:
    with pytest.raises(EngineError, match="unknown engine"):
        create_engine("nllb")


def test_create_engine_leanmt_reserved() -> None:
    with pytest.raises(EngineError, match="NEU-50"):
        create_engine("leanmt")
