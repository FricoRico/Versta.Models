"""
Typed payload shapes crossing the probe package boundaries: the on-disk
probe corpus format and the emitted scorecard JSON. Keep these in sync with
what ``scorecard.py`` actually writes — the scorecard schema is
manifest-adjacent and is agreed with the pipeline pair (review on the PR).
"""

from typing import List, Optional, TypedDict


class CorpusEntry(TypedDict):
    """One probe sentence with its human reference translation."""

    id: str
    tags: List[str]
    source: str
    reference: str


class Corpus(TypedDict):
    """One direction of the traveler probe corpus."""

    name: str
    source_language: str
    target_language: str
    version: int
    chrf_floor: float
    chrf_floor_note: str
    entries: List[CorpusEntry]


class ProbeSentence(TypedDict):
    """One translated probe sentence as reported by leanmt-probe."""

    source: str
    target: str
    ms: float
    ok: bool


class ProbeLatency(TypedDict):
    n: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class DecodeOptions(TypedDict):
    beam: int
    max_seq: int


class ProbeOutput(TypedDict, total=False):
    """The JSON leanmt-probe prints for one direction."""

    tool: str
    arch: str
    threads: int
    decode: DecodeOptions
    load_ok: bool
    load_error: str
    model_load_ms: float
    rss_after_load_kib: int
    sentences: List[ProbeSentence]
    latency: ProbeLatency
    rss_peak_kib: int


class DigitPreservation(TypedDict):
    passed: int
    total: int
    failing_ids: List[str]


class DirectionScore(TypedDict, total=False):
    """Per-direction measurement inside a scorecard."""

    direction: str
    load_ok: bool
    load_error: str
    model_load_ms: float
    rss_after_load_kib: int
    chrf: float
    chrf_floor: float
    direction_check: str  # "pass" | "fail"
    digit_preservation: DigitPreservation
    latency_ms: ProbeLatency
    rss_peak_kib: int
    sentences: int
    translation_failures: int


class ScorecardBundle(TypedDict):
    file: str
    sha256: str
    leanmt_binary_version: int
    bundle_version: str


class ScorecardEnvironment(TypedDict):
    host_arch: str
    probe_arch: str
    threads: int
    runner: str  # e.g. "ubuntu-24.04-arm" in CI, "local" elsewhere
    leanmt_commit: Optional[str]
    corpus: str
    corpus_entries: int


class Gates(TypedDict):
    """Which measurements gate this scorecard's acceptance verdict."""

    load: bool
    direction_chrf_floor: bool
    digit_preservation: bool
    latency_rss: bool


class Scorecard(TypedDict):
    """One bundle's acceptance scorecard; committed per pair under scorecards/."""

    scorecard_version: int
    slug: str
    generated_at: str
    bundle: ScorecardBundle
    environment: ScorecardEnvironment
    directions: List[DirectionScore]
    gates: Gates
    acceptance: str  # "pass" | "fail"
    expectation: str  # "pass" | "direction-fail"
    expectation_met: bool
