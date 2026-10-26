"""
Typed payload shapes crossing the probe package boundaries: the on-disk
probe corpus format and the emitted scorecard JSON. Keep these in sync with
what ``scorecard.py`` actually writes — the scorecard schema is
manifest-adjacent and is agreed with the pipeline pair (review on the PR).
"""

from typing import Dict, List, Optional, TypedDict

# Named schema string for the scorecard document. Every pipeline contract
# dispatches on a named string (``versta.eval-result/1``,
# ``versta.eval-corpora/1``, the loader manifests) — position-in-file is not
# a type. Bump on incompatible shape changes, in lockstep with
# ``SCORECARD_VERSION``.
SCORECARD_SCHEMA = "versta.pack-scorecard/1"

# Named schema string for the golden-band document pinned under
# ``pack-acceptance/bands.json`` (NEU-60). Manifest-adjacent like the
# scorecard schema: agreed with the pipeline pair before it lands.
BANDS_SCHEMA = "versta.pack-bands/1"


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


class ScorecardMetrics(TypedDict):
    """
    Provenance of the metric the card's chrF numbers are pinned to. chrF is
    scored by the stdlib re-implementation (``metrics.py``) and parity-tested
    against the installed sacrebleu; recording the library's verbatim
    signature + version pins case/whitespace handling and parameterization
    exactly, and makes any dependency drift visible per card.
    """

    sacrebleu_version: str
    chrf_signature: str  # verbatim CHRF.get_signature()


class ScorecardEnvironment(TypedDict):
    host_arch: str
    probe_arch: str
    threads: int
    runner: str  # e.g. "ubuntu-24.04-arm" in CI, "local" elsewhere
    leanmt_commit: Optional[str]
    corpus: str
    corpus_entries: int
    corpus_sha256: str  # payload hash over the corpus files scored against


class Gates(TypedDict):
    """Which measurements gate this scorecard's acceptance verdict."""

    load: bool
    direction_chrf_floor: bool
    digit_preservation: bool
    latency_rss: bool


class BandBaseline(TypedDict):
    """Where a direction band's golden measurement was taken."""

    scorecard: str  # repo-relative path of the champion scorecard
    bundle_sha256: str  # sha256 of the (re-packed) bundle that was measured
    leanmt_commit: Optional[str]
    measured_on: str  # e.g. "ubuntu-24.04-arm (aarch64)"


class DirectionBand(TypedDict):
    """
    The pinned acceptable band for one direction's chrF, derived from the
    golden measurement on the previously published bundle.

    ``golden_chrf`` is the measurement; ``chrf_min`` is the publish-blocking
    lower edge (``golden_chrf - bands.chrf_max_drop``); ``chrf_floor`` is the
    corpus-carried direction-sanity floor kept alongside for context and
    defense in depth (a mislabeled pack collapses through both).
    """

    golden_chrf: float
    chrf_min: float
    chrf_floor: float
    baseline: BandBaseline


class Bands(TypedDict):
    """
    Golden-output bands for every shipped direction (NEU-60), pinned in
    ``pack-acceptance/bands.json`` and consumed by the publish gate in
    ``versta.batch`` and the weekly sweep. The document is derivable from
    the committed baseline scorecards via ``versta.probe.bands``; it moves
    only in dedicated re-pin commits (bench-baseline discipline, NEU-16/22/23).
    """

    schema: str
    bands_version: int
    corpus: str  # corpus name every banded card must have been scored against
    chrf_max_drop: float  # publish-allowed absolute chrF drop below golden
    digit_preservation_required: float  # fraction; 1.0 = 100% (NEU-30)
    metrics: ScorecardMetrics
    directions: Dict[str, DirectionBand]
    note: str


class Scorecard(TypedDict):
    """
    One bundle's acceptance scorecard; committed per pair under scorecards/.

    Consumers dispatch on ``schema`` (``versta.pack-scorecard/1``) and join
    against the eval ledger (``versta.eval-result/1``) on
    ``bundle.sha256`` (lowercase hex both sides).
    """

    schema: str
    scorecard_version: int
    slug: str
    generated_at: str
    bundle: ScorecardBundle
    environment: ScorecardEnvironment
    metrics: ScorecardMetrics
    directions: List[DirectionScore]
    gates: Gates
    acceptance: str  # "pass" | "fail"
    expectation: str  # "pass" | "direction-fail"
    expectation_met: bool
