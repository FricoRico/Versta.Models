"""
Scorecard assembly and the acceptance rules for one probed bundle.

One scorecard JSON per bundle, committed per pair under ``scorecards/``
(first sweep pins the baselines) and re-emitted on every CI run as an
artifact. The v1 acceptance rules, per the approved NEU-45 proposal:

* every direction must load (``load_ok``);
* chrF against the corpus references must clear the per-direction floor
  carried by the corpus (the direction sanity check; a mislabeled-direction
  pack — the NEU-19 class — collapses below it);
* digit preservation must be 100% (the NEU-30 harm class);
* latency (p50/p95) and peak RSS are measured and published, but carry no
  v1 gate — absolute budgets are set by the Phase-4 device pass.

The caller passes an ``expectation``: ``pass`` for a real pack, or
``direction-fail`` for the adversarial negative control, where acceptance
must be a direction-check failure (any other failure kind, or an unexpected
pass, is itself a harness defect and fails the run).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import sacrebleu

from sacrebleu.metrics import CHRF

from .config import SCORECARD_VERSION

from .corpus import Corpus
from .install import InstalledBundle
from .metrics import corpus_chrf, digit_preservation
from .typing import (
    SCORECARD_SCHEMA,
    DigitPreservation,
    DirectionScore,
    Gates,
    ProbeOutput,
    Scorecard,
    ScorecardBundle,
    ScorecardEnvironment,
    ScorecardMetrics,
)

EXPECT_PASS = "pass"
EXPECT_DIRECTION_FAIL = "direction-fail"


def score_direction(
    direction_key: str, corpus: Corpus, output: ProbeOutput
) -> DirectionScore:
    """
    Scores one direction's probe output against its corpus.

    Args:
        direction_key (str): "<src>-<tgt>" of the direction.
        corpus (Corpus): the validated probe corpus for this direction.
        output (ProbeOutput): parsed leanmt-probe JSON.

    Returns:
        DirectionScore: load verdict, chrF + floor verdict, digit
            preservation and latency/RSS numbers.
    """
    if not output.get("load_ok", False):
        return DirectionScore(
            direction=direction_key,
            load_ok=False,
            load_error=output.get("load_error", "unknown load failure"),
            direction_check="fail",
        )

    sentences = output["sentences"]
    translation_failures = sum(1 for sentence in sentences if not sentence["ok"])

    source_by_id = {entry["source"]: entry for entry in corpus["entries"]}
    hypotheses: List[str] = []
    references: List[str] = []
    digit_fails: List[str] = []
    for sentence in sentences:
        entry = source_by_id.get(sentence["source"])
        if entry is None or not sentence["ok"]:
            continue
        hypotheses.append(sentence["target"])
        references.append(entry["reference"])
        if not digit_preservation(sentence["source"], sentence["target"]):
            digit_fails.append(entry["id"])

    chrf = corpus_chrf(hypotheses, references) if hypotheses else 0.0
    floor = float(corpus["chrf_floor"])

    return DirectionScore(
        direction=direction_key,
        load_ok=True,
        model_load_ms=output["model_load_ms"],
        rss_after_load_kib=output["rss_after_load_kib"],
        chrf=round(chrf, 2),
        chrf_floor=floor,
        direction_check="pass" if chrf >= floor else "fail",
        digit_preservation=DigitPreservation(
            passed=len(hypotheses) - len(digit_fails),
            total=len(hypotheses),
            failing_ids=digit_fails,
        ),
        latency_ms=output["latency"],
        rss_peak_kib=output["rss_peak_kib"],
        sentences=len(sentences),
        translation_failures=translation_failures,
    )


# Which measurements gate the v1 acceptance verdict. Per the approved NEU-45
# phasing, v1 gates are the two publish-blocking defect classes: loads (a pack
# that cannot load is dead) and the direction check (the NEU-19 class). Digit
# preservation (NEU-30) and latency/RSS are measured and published from the
# first sweep on and become publish gates in NEU-62 once per-pair bands are
# calibrated from measured data — gating v1 on uncalibrated thresholds would
# cry wolf.
GATING_V1 = Gates(
    load=True,
    direction_chrf_floor=True,
    digit_preservation=False,
    latency_rss=False,
)


def metric_provenance() -> ScorecardMetrics:
    """
    Records the exact chrF parameterization the card's numbers are pinned
    to: the installed sacrebleu's version and its verbatim
    ``CHRF.get_signature()``. The probe scores chrF with the stdlib
    re-implementation in ``metrics.py``; the test suite asserts parity
    against this library's ``corpus_score`` on pinned inputs, so the
    library's signature describes the scored numbers exactly — including
    case and whitespace handling, which a shorthand would lose.

    Returns:
        ScorecardMetrics: sacrebleu version + verbatim chrF signature.
    """
    metric = CHRF()
    # get_signature() resolves nrefs only after an evaluation; probe corpora
    # are single-reference, so scoring one pinned pair yields nrefs:1.
    metric.corpus_score(["probe"], [["probe"]])
    return ScorecardMetrics(
        sacrebleu_version=sacrebleu.__version__,
        chrf_signature=str(metric.get_signature()),
    )


def acceptance_of(
    directions: List[DirectionScore],
) -> str:
    """
    Reduces per-direction scores to the bundle acceptance verdict under the
    v1 gates (``GATING_V1``): every direction must load and clear its chrF
    direction floor. Digit preservation, latency and RSS are always measured
    and recorded; they gate from NEU-62 on calibrated per-pair bands.

    Args:
        directions (List[DirectionScore]): one entry per probed direction.

    Returns:
        str: "pass" or "fail".
    """
    for score in directions:
        if not score["load_ok"]:
            return "fail"
        if score["direction_check"] != "pass":
            return "fail"
    return "pass"


def build_scorecard(
    bundle: InstalledBundle,
    direction_scores: List[DirectionScore],
    environment: ScorecardEnvironment,
    expectation: str,
    generated_at: Optional[datetime] = None,
) -> Scorecard:
    """
    Assembles the final scorecard for one probed bundle.

    Args:
        bundle (InstalledBundle): the installed bundle (identity + hashes).
        direction_scores (List[DirectionScore]): per-direction measurements.
        environment (ScorecardEnvironment): runner/host provenance.
        expectation (str): "pass" or "direction-fail".
        generated_at (Optional[datetime]): override for reproducibility in
            tests; defaults to now UTC.

    Returns:
        Scorecard: the complete scorecard with acceptance + expectation_met.
    """
    acceptance = acceptance_of(direction_scores)
    # The adversarial control must trip the direction check *on a pack that
    # loaded fine* — a load failure proves the load gate, not the chrF
    # direction gate the control exists to verify.
    direction_fail_after_load = any(
        score.get("load_ok", False) and score["direction_check"] == "fail"
        for score in direction_scores
    )
    if expectation == EXPECT_PASS:
        met = acceptance == "pass"
    else:
        met = acceptance == "fail" and direction_fail_after_load

    stamp = generated_at or datetime.now(timezone.utc)
    return Scorecard(
        schema=SCORECARD_SCHEMA,
        scorecard_version=SCORECARD_VERSION,
        generated_at=stamp.isoformat(timespec="seconds"),
        slug=bundle.slug,
        bundle=ScorecardBundle(
            file=bundle.archive.name,
            sha256=bundle.sha256,
            leanmt_binary_version=bundle.leanmt_binary_version,
            bundle_version=bundle.bundle_version,
        ),
        environment=environment,
        metrics=metric_provenance(),
        directions=direction_scores,
        gates=GATING_V1,
        acceptance=acceptance,
        expectation=expectation,
        expectation_met=met,
    )


def write_scorecard(scorecard: Scorecard, output_path: Path) -> None:
    """
    Writes a scorecard as stable, diff-friendly JSON (2-space indent, key
    order as constructed, trailing newline).

    Args:
        scorecard (Scorecard): the assembled scorecard.
        output_path (Path): destination file.
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
