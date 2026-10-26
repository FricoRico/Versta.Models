"""
Orchestration for one pack-acceptance run: install the bundle, probe every
declared direction against its corpus, score, emit the scorecard.
"""

import os
import platform

from pathlib import Path
from typing import Optional

from ..schema import ValidationError
from .bridge import run_probe
from .config import CORPUS_NAME, DEFAULT_CORPUS_DIR, ProbeConfig
from .corpus import corpus_path_for, corpus_payload_sha256, load_corpus
from .install import InstalledBundle, install_bundle
from .scorecard import build_scorecard, score_direction, write_scorecard
from .typing import Scorecard, ScorecardEnvironment


class RunnerError(RuntimeError):
    """The run could not proceed (harness setup problem, not a pack verdict)."""


def run_bundle_probe(
    bundle_path: Path,
    probe_bin: Path,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = Path("output/probe"),
    staging_dir: Optional[Path] = None,
    threads: Optional[int] = None,
    expectation: str = "pass",
    label: Optional[str] = None,
    leanmt_commit: Optional[str] = None,
) -> Scorecard:
    """
    Installs and probes one bundle end to end and writes its scorecard.

    Args:
        bundle_path (Path): bundle tarball (``.sha256`` sidecar verified when
            present next to it — CI always stages it).
        probe_bin (Path): built leanmt-probe binary.
        corpus_dir (Path): corpus directory; defaults to the packaged
            traveler-v1 corpora.
        output_dir (Path): where scorecards and probe JSON are written.
        staging_dir (Optional[Path]): bundle extraction scratch; defaults to
            ``output_dir/install``.
        threads (Optional[int]): intra-op thread override for leanmt-probe.
        expectation (str): "pass" for real packs, "direction-fail" for the
            adversarial control (acceptance must then fail *because* of the
            direction check).
        label (Optional[str]): scorecard filename stem override; defaults to
            the bundle's language slug, with ".adversarial" appended when
            expectation is "direction-fail".
        leanmt_commit (Optional[str]): provenance for the scorecard's
            environment block; CI records the pinned commit it built.

    Returns:
        Scorecard: the assembled and written scorecard.

    Raises:
        RunnerError: corpus missing for a declared direction, no directions
            in the bundle, or the probe binary unavailable; ValidationError
            (versta.schema) when the bundle fails the contract gate.
    """
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    bundle: InstalledBundle = install_bundle(
        bundle_path, staging_dir or (output_dir / "install")
    )
    if not bundle.directions:
        raise RunnerError(f"{bundle_path.name}: bundle declares no directions")

    config = ProbeConfig(threads=threads or ProbeConfig().threads)
    direction_scores = []
    corpus_entries = 0
    corpus_paths = []
    probe_arch = "unknown"
    for direction in bundle.directions:
        direction_key = f"{direction.source_language}-{direction.target_language}"
        corpus_path = corpus_path_for(
            corpus_dir, direction.source_language, direction.target_language
        )
        if not corpus_path.is_file():
            raise RunnerError(
                f"no probe corpus for declared direction {direction_key} "
                f"(looked in {corpus_dir}); the harness cannot score a "
                "direction without references"
            )
        corpus = load_corpus(corpus_path)
        if corpus["name"] != CORPUS_NAME:
            raise RunnerError(
                f"corpus {corpus_path.name} is '{corpus['name']}', this runner "
                f"scores '{CORPUS_NAME}' — re-baseline deliberately rather "
                "than mixing corpus versions"
            )
        corpus_entries += len(corpus["entries"])
        corpus_paths.append(corpus_path)

        output = run_probe(
            probe_bin,
            direction,
            [entry["source"] for entry in corpus["entries"]],
            work_dir,
            config=config,
        )
        probe_arch = output.get("arch", "unknown")
        direction_scores.append(score_direction(direction_key, corpus, output))

    environment = ScorecardEnvironment(
        host_arch=platform.machine(),
        probe_arch=probe_arch,
        threads=int(output.get("threads", 0)) if direction_scores else 0,
        runner=os.environ.get("PACK_ACCEPTANCE_RUNNER", "local"),
        leanmt_commit=leanmt_commit or os.environ.get("PACK_ACCEPTANCE_LEANMT_COMMIT"),
        corpus=CORPUS_NAME,
        corpus_entries=corpus_entries,
        corpus_sha256=corpus_payload_sha256(corpus_paths),
    )
    scorecard = build_scorecard(
        bundle,
        direction_scores,
        environment,
        expectation=expectation,
    )

    stem = label or (
        bundle.slug + (".adversarial" if expectation == "direction-fail" else "")
    )
    write_scorecard(scorecard, output_dir / "scorecards" / f"{stem}.json")
    return scorecard


def summarize(scorecard: Scorecard) -> str:
    """
    Renders one scorecard as a compact terminal/CI table.

    Args:
        scorecard (Scorecard): the scorecard to render.

    Returns:
        str: multi-line summary text.
    """
    lines = [
        f"pack {scorecard['slug']} ({scorecard['bundle']['file']}, "
        f"sha256 {scorecard['bundle']['sha256'][:12]}…)",
    ]
    for direction in scorecard["directions"]:
        if not direction["load_ok"]:
            lines.append(
                f"  {direction['direction']}: LOAD FAIL — {direction['load_error']}"
            )
            continue
        digits = direction["digit_preservation"]
        latency = direction["latency_ms"]
        lines.append(
            f"  {direction['direction']}: load {direction['model_load_ms']:.0f} ms, "
            f"chrF {direction['chrf']:.1f} (floor {direction['chrf_floor']}) "
            f"-> {direction['direction_check']}, "
            f"digits {digits['passed']}/{digits['total']}, "
            f"p50 {latency['p50_ms']:.0f} ms / p95 {latency['p95_ms']:.0f} ms, "
            f"RSS peak {direction['rss_peak_kib'] / 1024:.0f} MiB"
        )
    lines.append(
        f"acceptance: {scorecard['acceptance']} | expectation: "
        f"{scorecard['expectation']} -> "
        + ("OK" if scorecard["expectation_met"] else "UNEXPECTED OUTCOME")
    )
    return "\n".join(lines)


__all__ = ["RunnerError", "ValidationError", "run_bundle_probe", "summarize"]
