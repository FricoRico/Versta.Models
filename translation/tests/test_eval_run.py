import importlib.util
import json

from hashlib import sha256
from pathlib import Path

import pytest

from versta.eval.__main__ import main
from versta.eval.engines import BergamotEngine, EngineError
from versta.eval.evaluate import evaluate_direction
from versta.eval.typing import EVAL_RESULT_SCHEMA, EvalResult

from conftest import (
    EVAL_PAIRS,
    write_direction_dir,
    write_eval_bundle,
    write_eval_corpus,
    write_fake_bergamot,
)


def _evaluate(
    tmp_path: Path,
    binary: Path,
    *,
    corpus_manifest: Path,
    bundle: bool = False,
) -> EvalResult:
    engine = BergamotEngine(binary=binary)
    if bundle:
        bundle_path = write_eval_bundle(tmp_path / "bundle")
        return evaluate_direction(
            source="en",
            target="es",
            engine=engine,
            corpus_manifest_path=corpus_manifest,
            bundle=bundle_path,
            output_path=tmp_path / "result.json",
        )
    direction_dir = write_direction_dir(
        tmp_path / "in", "en-es", model_name="model.enes.intgemm.alphas.bin"
    )
    return evaluate_direction(
        source="en",
        target="es",
        engine=engine,
        corpus_manifest_path=corpus_manifest,
        direction_dir=direction_dir,
        output_path=tmp_path / "result.json",
    )


def test_eval_end_to_end_scores_fixture_corpus(
    tmp_path: Path, fake_bergamot: Path
) -> None:
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    result = _evaluate(tmp_path / "b", fake_bergamot, corpus_manifest=corpus_manifest)

    assert result["schema"] == EVAL_RESULT_SCHEMA
    assert result["direction"] == "en-es"
    assert result["scores"] == {"bleu": 100.0, "chrf": 100.0, "comet22": None}
    assert result["corpus"]["revision"] == "tests/fixture@0001"
    assert result["corpus"]["sentences"] == len(EVAL_PAIRS)
    assert result["metrics"]["sacrebleu_version"]
    assert result["engine"]["name"] == "bergamot-translator"
    assert result["engine"]["variant"] == "local"
    assert result["engine"]["version"] == "bergamot-fake 0.0.0-test"
    # Not a bundle run: no bundle fields, but the model hash is recorded.
    assert "bundle_sha256" not in result["model"]
    assert len(result["model"]["model_sha256"]) == 64

    written = json.loads((tmp_path / "b" / "result.json").read_text("utf-8"))
    assert written == result


def test_eval_reproducible_on_rerun(tmp_path: Path, fake_bergamot: Path) -> None:
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    first = _evaluate(tmp_path / "r1", fake_bergamot, corpus_manifest=corpus_manifest)
    second = _evaluate(tmp_path / "r2", fake_bergamot, corpus_manifest=corpus_manifest)

    # Same inputs -> same scores and same scored hypothesis; only wall-clock
    # fields may differ between runs.
    assert first["scores"] == second["scores"]
    assert first["run"]["hypothesis_sha256"] == second["run"]["hypothesis_sha256"]
    assert first["engine"]["config_sha256"] == second["engine"]["config_sha256"]


def test_eval_from_shipped_bundle_records_bundle_checksum(
    tmp_path: Path, fake_bergamot: Path
) -> None:
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    result = _evaluate(
        tmp_path / "b", fake_bergamot, corpus_manifest=corpus_manifest, bundle=True
    )

    assert result["scores"]["bleu"] == 100.0
    assert result["model"]["bundle_file"] == "en-es-bundle.tar.gz"
    bundle_sha256 = result["model"]["bundle_sha256"]

    actual = sha256(
        (tmp_path / "b" / "bundle" / "en-es-bundle.tar.gz").read_bytes()
    ).hexdigest()
    assert bundle_sha256 == actual


def test_eval_rejects_direction_mismatch(tmp_path: Path, fake_bergamot: Path) -> None:
    # The corpus covers the requested direction; the artifact must still
    # match it before any engine time is spent.
    corpus_manifest = write_eval_corpus(tmp_path / "a", direction="es-en")
    direction_dir = write_direction_dir(
        tmp_path / "in", "en-es", model_name="model.enes.intgemm.alphas.bin"
    )
    with pytest.raises(ValueError, match="wrong direction"):
        evaluate_direction(
            source="es",
            target="en",
            engine=BergamotEngine(binary=fake_bergamot),
            corpus_manifest_path=corpus_manifest,
            direction_dir=direction_dir,
            output_path=tmp_path / "result.json",
        )


def test_eval_requires_exactly_one_artifact(
    tmp_path: Path, fake_bergamot: Path
) -> None:
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_direction(
            source="en",
            target="es",
            engine=BergamotEngine(binary=fake_bergamot),
            corpus_manifest_path=corpus_manifest,
            output_path=tmp_path / "result.json",
        )


def test_eval_surfaces_engine_contract_violations(tmp_path: Path) -> None:
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    misaligned = write_fake_bergamot(tmp_path / "bergamot", mode="drop-last")
    with pytest.raises(EngineError, match="not line-aligned"):
        _evaluate(tmp_path / "b", misaligned, corpus_manifest=corpus_manifest)
    # No result JSON may be written on a failed eval.
    assert not (tmp_path / "b" / "result.json").exists()


def test_digit_corrupting_engine_scores_measurably_lower(
    tmp_path: Path, fake_bergamot: Path
) -> None:
    """
    NEU-30-class guard: an engine that corrupts digits (the digit-forcing
    shortlist defect) must lose real points against the pinned corpus — the
    harness scores translation output, not packaging.
    """
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    faithful = _evaluate(
        tmp_path / "good", fake_bergamot, corpus_manifest=corpus_manifest
    )
    mangling = write_fake_bergamot(tmp_path / "bergamot-bad", mode="digit-mangle")
    defective = _evaluate(tmp_path / "bad", mangling, corpus_manifest=corpus_manifest)

    assert defective["scores"]["chrf"] < faithful["scores"]["chrf"] - 1.0
    assert defective["scores"]["bleu"] < faithful["scores"]["bleu"]
    # And the runs provably scored different translations.
    assert defective["run"]["hypothesis_sha256"] != faithful["run"]["hypothesis_sha256"]


def test_comet_flag_requires_optional_dependency(
    tmp_path: Path, fake_bergamot: Path
) -> None:
    if importlib.util.find_spec("comet") is not None:
        pytest.skip("unbabel-comet installed in this environment")
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    with pytest.raises(EngineError, match="unbabel-comet"):
        direction_dir = write_direction_dir(
            tmp_path / "in", "en-es", model_name="model.enes.intgemm.alphas.bin"
        )
        evaluate_direction(
            source="en",
            target="es",
            engine=BergamotEngine(binary=fake_bergamot),
            corpus_manifest_path=corpus_manifest,
            direction_dir=direction_dir,
            with_comet=True,
            output_path=tmp_path / "result.json",
        )


def test_main_resolves_engine_from_env(
    tmp_path: Path, fake_bergamot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_manifest = write_eval_corpus(tmp_path / "a")
    monkeypatch.setenv("VERSTA_BERGAMOT_BIN", str(fake_bergamot))
    direction_dir = write_direction_dir(
        tmp_path / "in", "en-es", model_name="model.enes.intgemm.alphas.bin"
    )
    output = tmp_path / "result.json"
    written = main(
        source="en",
        target="es",
        direction_dir=direction_dir,
        bundle=None,
        corpus_manifest=corpus_manifest,
        engine="bergamot",
        bergamot_bin=None,
        bergamot_image=None,
        comet=False,
        output=output,
        keep_workdir=False,
    )
    assert written == output
    result = json.loads(output.read_text("utf-8"))
    assert result["scores"]["bleu"] == 100.0
