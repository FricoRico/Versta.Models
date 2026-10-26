"""
TypedDict contracts for the values versta.eval reads and writes.

One document crosses the module boundary:

  * the eval result JSON (schema ``versta.eval-result/1``) — written once
    per evaluated direction, consumed by the release gate and the
    ``evals/`` ledger (NEU-52), so the shape is additive-only.

The corpus manifest the harness reads (``evals/corpora/flores200-dev.json``,
schema ``versta/corpus-manifest@1``) is typed in
``versta.eval.corpora.typing`` — the corpus package owns its contract, the
harness consumes it read-only.

Keep both fully typed and in sync with the checked-in schema
documentation in ``evals/README.md``.
"""

from typing import Dict, Optional, TypedDict

EVAL_RESULT_SCHEMA = "versta.eval-result/1"


class ModelInfo(TypedDict, total=False):
    base_model: str
    version: str
    architecture: str
    model_sha256: str
    metadata_sha256: str
    bundle_file: str
    bundle_sha256: str


class CorpusInfo(TypedDict):
    corpus: str
    split: str
    revision: str
    sentences: int
    source_sha256: str
    target_sha256: str


class EngineInfo(TypedDict, total=False):
    name: str
    variant: str
    version: str
    image: str
    decode: Dict[str, object]
    config_sha256: str


class MetricsInfo(TypedDict):
    sacrebleu_version: str
    bleu_signature: str
    chrf_signature: str
    comet22_model: Optional[str]


class Scores(TypedDict):
    bleu: float
    chrf: float
    comet22: Optional[float]


class RunInfo(TypedDict):
    started_utc: str
    duration_seconds: float
    hypothesis_sha256: str


class EvalResult(TypedDict):
    schema: str
    direction: str
    source_language: str
    target_language: str
    model: ModelInfo
    corpus: CorpusInfo
    engine: EngineInfo
    metrics: MetricsInfo
    scores: Scores
    run: RunInfo
