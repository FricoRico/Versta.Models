"""
TypedDict contracts for the values versta.eval reads and writes.

Three documents cross the module boundary:

  * the corpus manifest (``evals/corpora.json``, schema
    ``versta.eval-corpora/1``) — produced by the corpus pinning task
    (NEU-49), consumed read-only here;
  * the eval result JSON (schema ``versta.eval-result/1``) — written once
    per evaluated direction, consumed by the release gate and the
    ``evals/`` ledger (NEU-52), so the shape is additive-only;
  * the direction ledger (``evals/ledger/<direction>.json``, schema
    ``versta.eval-ledger/1``) — the checked-in per-direction history of
    shipped versions, written by the release gate's ``--record`` path and
    read back as the predecessor source for later refreshes.

Keep all of them fully typed and in sync with the checked-in schema
documentation in ``evals/README.md``.
"""

from typing import Dict, List, Optional, TypedDict

# Corpus manifest (input, produced by the corpus pinning task).

CORPORA_SCHEMA = "versta.eval-corpora/1"


class CorpusFile(TypedDict):
    path: str
    sha256: str


class CorpusDirection(TypedDict):
    flores_source: str
    flores_target: str
    sentences: int
    source: CorpusFile
    target: CorpusFile


class CorpusManifest(TypedDict):
    schema: str
    corpus: str
    revision: str
    split: str
    license: str
    directions: Dict[str, CorpusDirection]


# Eval result (output, consumed by the release gate and ledger).

EVAL_RESULT_SCHEMA = "versta.eval-result/1"


class ModelInfo(TypedDict, total=False):
    base_model: str
    version: str
    architecture: str
    model_sha256: str
    metadata_sha256: str
    bundle_file: str
    bundle_sha256: str
    # Bytes of the artifact under evaluation: the tarball for bundle runs,
    # the sum of model/vocabulary/shortlist payloads for direction-dir runs.
    # Recorded so the ledger and the generated eval summary can carry size
    # notes without touching the filesystem.
    size_bytes: int


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


# Direction ledger (checked-in, append-only; one file per direction).

LEDGER_SCHEMA = "versta.eval-ledger/1"

# Gate decisions that can be recorded in the ledger. "fail" never lands in
# the ledger: a blocked refresh leaves history untouched by construction.
#   baseline   — first recorded version for the direction (no predecessor)
#   pass       — chrF regression within threshold vs. the predecessor
#   override   — regression beyond threshold, shipped with a written reason
#   rebaseline — corpus pin changed vs. the predecessor, so scores are
#                non-comparable by construction; shipped with a written reason
GATE_DECISIONS = ("baseline", "pass", "override", "rebaseline")


class GateInfo(TypedDict):
    decision: str
    threshold_chrf: float
    predecessor_version: Optional[str]
    delta_bleu: Optional[float]
    delta_chrf: Optional[float]
    delta_comet22: Optional[float]
    reason: Optional[str]
    decided_utc: str


class LedgerEntry(TypedDict):
    version: str
    result: EvalResult
    gate: GateInfo


class DirectionLedger(TypedDict):
    schema: str
    direction: str
    source_language: str
    target_language: str
    entries: List[LedgerEntry]
