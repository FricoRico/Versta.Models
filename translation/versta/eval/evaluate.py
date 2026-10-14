"""
Eval orchestration: corpus → engine → metrics → result JSON.

One call of `evaluate_direction` evaluates exactly one direction: it
verifies the pinned corpus slice, resolves the produced artifact (a
``versta.download`` direction directory or an extracted shipped bundle),
translates the corpus through the reference engine, scores sacreBLEU and
chrF (COMET-22 only when requested), and writes the versioned result JSON
the release gate and ``evals/`` ledger consume.

Reproducibility contract: for a fixed artifact, corpus revision, engine
and decode config, a re-run must produce identical scores. Everything the
score depends on — bundle/model checksums, corpus payload hashes, engine
version and variant, the rendered engine config hash, the metric
signatures and the hypothesis hash — is recorded in the result, so a
mismatch anywhere is visible rather than silent.
"""

import json
import os
import shutil
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import corpora
from .comet import COMET22_MODEL, score_comet22
from .engines import Engine
from .metrics import score_translations
from .models import load_direction_dir, load_from_bundle, sha256_file
from .typing import (
    EVAL_RESULT_SCHEMA,
    CorpusInfo,
    EngineInfo,
    EvalResult,
    MetricsInfo,
    ModelInfo,
    RunInfo,
    Scores,
)

DEFAULT_OUTPUT_DIR = Path("output") / "evals"


def write_result(result: EvalResult, output_path: Path) -> Path:
    """
    Writes the result JSON atomically (sorted keys, trailing newline) so
    half-written results can never be picked up by the gate.

    Args:
        result (EvalResult): The completed eval result.
        output_path (Path): Destination file.

    Returns:
        Path: The written path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    os.replace(temp_path, output_path)
    return output_path


DEFAULT_CORPUS_ROOT = Path("output") / "corpora"


def evaluate_direction(
    *,
    source: str,
    target: str,
    engine: Engine,
    corpus_manifest_path: Path,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    direction_dir: Optional[Path] = None,
    bundle: Optional[Path] = None,
    with_comet: bool = False,
    output_path: Optional[Path] = None,
    work_root: Optional[Path] = None,
    keep_workdir: bool = False,
) -> EvalResult:
    """
    Evaluates one direction end to end and writes the result JSON.

    Exactly one of ``direction_dir`` / ``bundle`` selects the artifact under
    evaluation. The corpus slice is verified against the pinned manifest
    before the engine starts, and the engine's output is required to be
    line-aligned with the corpus — any violation raises, and no result is
    written.

    Args:
        source (str): Source language code (e.g. ``en``).
        target (str): Target language code (e.g. ``es``).
        engine (Engine): The configured reference runtime.
        corpus_manifest_path (Path): The pinned corpus manifest
            (``evals/corpora/flores200-dev.json``).
        corpus_root (Path): The fetch cache root the verified snapshot is
            read from (``output/corpora`` by default; run
            ``uv run python -m versta.eval.corpora fetch`` first).
        direction_dir (Path | None): A ``versta.download`` output directory.
        bundle (Path | None): A shipped ``*-bundle.tar.gz`` to evaluate from.
        with_comet (bool): Also score COMET-22 (release tier; needs the
            optional unbabel-comet dependency and cached weights).
        output_path (Path | None): Result JSON destination; defaults to
            ``output/evals/<source>-<target>.json``.
        work_root (Path | None): Parent for the engine work directory
            (defaults to the system temp dir).
        keep_workdir (bool): Keep the work directory (inputs, engine config,
            raw hypotheses) for debugging instead of cleaning it up.

    Returns:
        EvalResult: The completed result (also written to ``output_path``).

    Raises:
        ValueError: On invalid argument combinations.
        CorpusError: If the corpus slice fails verification.
        ModelInputError: If the artifact cannot be resolved.
        ValidationError: If artifact metadata violates the app schema.
        EngineError: On engine failure or a line-alignment violation.
    """
    if (direction_dir is None) == (bundle is None):
        raise ValueError(
            "pass exactly one of direction_dir (pre-bundling artifact) or "
            "bundle (shipped artifact)"
        )

    direction = f"{source}-{target}"
    started = datetime.now(timezone.utc)

    # Verify the pinned corpus before spending any engine time.
    manifest = corpora.load_manifest(corpus_manifest_path)
    corpus = corpora.resolve_slice(
        manifest, direction, corpus_dir=corpora.snapshot_dir(manifest, corpus_root)
    )

    work_dir = Path(tempfile.mkdtemp(prefix="versta-eval-", dir=work_root))
    try:
        if bundle is not None:
            model = load_from_bundle(bundle, direction, work_dir / "unpacked")
        else:
            assert direction_dir is not None  # enforced by the check above
            model = load_direction_dir(direction_dir)

        if model.source_language != source or model.target_language != target:
            raise ValueError(
                f"requested {direction} but the artifact carries "
                f"{model.direction} — refusing to attribute scores to the "
                f"wrong direction"
            )

        outcome = engine.translate(model, corpus.source_sentences, work_dir / "engine")

        metrics = score_translations(outcome.translations, corpus.target_sentences)

        comet_score: Optional[float] = None
        comet_model: Optional[str] = None
        if with_comet:
            comet_score = score_comet22(
                corpus.source_sentences,
                outcome.translations,
                corpus.target_sentences,
            )
            comet_model = COMET22_MODEL

        finished = datetime.now(timezone.utc)

        model_info = ModelInfo(
            base_model=model.base_model,
            version=model.version,
            architecture=model.architecture,
            model_sha256=model.model_sha256,
            metadata_sha256=model.metadata_sha256,
        )
        if model.bundle_sha256 is not None and model.bundle_file is not None:
            model_info["bundle_file"] = model.bundle_file
            model_info["bundle_sha256"] = model.bundle_sha256

        engine_info = EngineInfo(
            name=engine.name,
            variant=outcome.variant,
            version=outcome.version,
            decode=outcome.decode,
            config_sha256=outcome.config_sha256,
        )
        if outcome.image is not None:
            engine_info["image"] = outcome.image

        result = EvalResult(
            schema=EVAL_RESULT_SCHEMA,
            direction=direction,
            source_language=source,
            target_language=target,
            model=model_info,
            corpus=CorpusInfo(
                corpus=corpus.corpus,
                split=corpus.split,
                revision=corpus.revision,
                sentences=len(corpus.source_sentences),
                source_sha256=corpus.source_sha256,
                target_sha256=corpus.target_sha256,
            ),
            engine=engine_info,
            metrics=MetricsInfo(
                sacrebleu_version=metrics.sacrebleu_version,
                bleu_signature=metrics.bleu_signature,
                chrf_signature=metrics.chrf_signature,
                comet22_model=comet_model,
            ),
            scores=Scores(bleu=metrics.bleu, chrf=metrics.chrf, comet22=comet_score),
            run=RunInfo(
                started_utc=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                duration_seconds=round((finished - started).total_seconds(), 3),
                hypothesis_sha256=sha256_file(outcome.translation_path),
            ),
        )
    finally:
        if keep_workdir:
            print(f"Keeping eval work directory: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    destination = output_path or (DEFAULT_OUTPUT_DIR / f"{direction}.json")
    write_result(result, destination)
    return result
