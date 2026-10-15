"""
Typed payloads crossing function boundaries in the Supertonic 3 pipeline.

Conventions follow the module charter: TypedDicts everywhere, no loosely
typed dicts, and schema-stamped JSON documents for anything checked in.
"""

from typing import List, Literal, Optional, TypedDict

# JSON document schemas. Conversion records are versioned documents, same as
# the versta.eval ledger stream (NEU-49..54).
CONVERSION_LEDGER_SCHEMA = "versta.tts-conversion-ledger/1"
VERIFICATION_REPORT_SCHEMA = "versta.tts-verification/1"

ModelStage = Literal[
    "duration_predictor", "text_encoder", "vector_estimator", "vocoder"
]
Precision = Literal["fp32", "fp16", "int8"]

#: The four Supertonic 3 stages, in pipeline order.
MODEL_STAGES: List[ModelStage] = [
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
]

#: Precisions the pipeline produces. int8 ships; fp16/fp32 are gate references.
PRECISIONS: List[Precision] = ["fp32", "fp16", "int8"]


class SourcePin(TypedDict):
    """A single upstream file pinned by content hash."""

    path: str  # path inside the upstream repo, e.g. "onnx/tts.json"
    sha256: str  # expected content hash; verification fails closed on mismatch


class ArtifactRecord(TypedDict):
    """One produced artifact (a converted .mnn or a passthrough file)."""

    path: str  # path relative to the conversion output directory
    sha256: str
    bytes: int


class UpstreamRecord(TypedDict):
    """What the conversion consumed."""

    repo: str  # Hugging Face repo id, e.g. "Supertone/supertonic-3"
    revision: str  # pinned commit sha; never a moving ref


class ToolchainRecord(TypedDict):
    """What produced the artifacts (reproducibility)."""

    mnn_commit: str  # commit of vendor/mnn the MNNConvert binary was built from
    mnn_tag: str  # readable tag of the above, e.g. "3.6.1"
    python: str


class StageMetric(TypedDict):
    """Numerics for one stage at one precision vs the fp32 ONNX reference."""

    stage: ModelStage
    precision: Precision
    snr_db: float
    cosine: float
    # Wav-only (vocoder): log-spectral distance and spectral correlation.
    lsd_db: Optional[float]
    spectral_r: Optional[float]
    passed: bool


class VerificationReport(TypedDict):
    """Checked-in/PR-attached verification output (schema VERIFICATION_REPORT_SCHEMA)."""

    schema: str
    seed: int
    upstream: UpstreamRecord
    toolchain: ToolchainRecord
    metrics: List[StageMetric]
    thresholds_ref: str  # pointer to the threshold table used (docstring/table id)
    passed: bool


class LedgerEntry(TypedDict):
    """One shipped-version record in the conversion ledger."""

    recorded_at: str  # ISO-8601 UTC
    version: str  # bundle version, e.g. "v2.0.0"
    upstream: UpstreamRecord
    toolchain: ToolchainRecord
    artifacts: List[ArtifactRecord]
    verification: VerificationReport
    bundle_sha256: str
    bundle_bytes: int
    override_reason: Optional[str]  # non-empty only when a gate was overridden


class ConversionLedger(TypedDict):
    """Append-only checked-in history for the supertonic3 bundle."""

    schema: str
    model: str
    entries: List[LedgerEntry]


class VoiceRow(TypedDict):
    """One feed voice row; order and values pinned by the app-side feed test."""

    gender: str  # "female" | "male"
    language: str  # ISO code, e.g. "en"
