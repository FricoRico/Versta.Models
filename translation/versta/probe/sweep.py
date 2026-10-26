"""
Full-catalog pack acceptance sweep (NEU-59): probes every published bundle
through the same production path as single-bundle runs and writes one
scorecard per bundle plus a machine-readable sweep summary.

The published artifacts predate the current bundler contract (v2.0.0 packs
carry no ``leanmt_binary_version`` handshake, NEU-25), so the sweep first
re-bundles each payload through ``versta.bundle`` — the production path —
before probing. The published payload bytes themselves are what gets
measured; re-packing only re-stamps the manifest.
"""

import json
import shutil
import tarfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bundle.__main__ import main as bundle_main
from .install import sha256_of, verify_checksum_sidecar
from .runner import RunnerError, run_bundle_probe
from .typing import Scorecard

SWEEP_SUMMARY_SCHEMA = "versta.pack-sweep/1"


class SweepError(RuntimeError):
    """One bundle could not be swept; the sweep carries on and reports it."""


@dataclass
class BundleSweepResult:
    """Outcome of sweeping one published bundle."""

    slug: str
    published_file: str
    published_sha256: str
    repacked_sha256: Optional[str] = None
    scorecard_path: Optional[Path] = None
    scorecard: Optional[Scorecard] = None
    error: Optional[str] = None


@dataclass
class SweepReport:
    """Aggregate of one full sweep run."""

    results: List[BundleSweepResult] = field(default_factory=list)

    @property
    def failures(self) -> List[BundleSweepResult]:
        """Bundles that hard-failed (harness/install error) or failed acceptance."""
        return [
            result
            for result in self.results
            if result.error is not None
            or result.scorecard is None
            or not result.scorecard["expectation_met"]
        ]


def repack_published_bundle(
    published_gz: Path, work_dir: Path, repack_dir: Path
) -> Path:
    """
    Re-bundles the payload of one published pack through the production
    bundler, after verifying the published checksum. Needed because the live
    catalog ships v2.0.0 manifests that predate the current bundle contract.

    Args:
        published_gz (Path): the published ``<slug>-bundle.tar.gz``.
        work_dir (Path): scratch for extraction (cleared per bundle).
        repack_dir (Path): where the re-bundled tarball is written.

    Returns:
        Path: the re-bundled ``<slug>-bundle.tar.gz`` (with ``.sha256``
            sidecar next to it).

    Raises:
        SweepError: extraction or bundling failed — recorded, never raised
            past the sweep loop.
    """
    try:
        verify_checksum_sidecar(published_gz)
    except Exception as error:
        raise SweepError(f"{published_gz.name}: checksum gate failed: {error}")
    extract_dir = work_dir / "extract"
    if extract_dir.exists():
        for child in sorted(extract_dir.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
    extract_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(extract_dir.iterdir(), reverse=True):
        if child.exists():
            if child.is_dir():
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                    elif sub.is_dir():
                        sub.rmdir()
                child.rmdir()
            else:
                child.unlink()
    try:
        with tarfile.open(published_gz, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise SweepError(f"{published_gz.name}: cannot extract: {error}")

    directions = sorted(
        child
        for child in extract_dir.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    )
    if not directions:
        raise SweepError(f"{published_gz.name}: no direction payload directories")
    result = bundle_main(
        input_dirs=directions,
        output_dir=repack_dir,
        bidirectional=len(directions) > 1,
        keep_intermediates=False,
        keep_input=True,
    )
    return result["bundle"]


def sweep_bundles(
    bundles_dir: Path,
    probe_bin: Path,
    output_dir: Path,
    corpus_dir: Optional[Path] = None,
    threads: Optional[int] = None,
    leanmt_commit: Optional[str] = None,
) -> SweepReport:
    """
    Sweeps every published bundle in ``bundles_dir``: verify checksum,
    re-pack through ``versta.bundle``, probe, scorecard. Never stops on a
    single failure; failures are collected and reported at the end.

    Args:
        bundles_dir (Path): published ``*-bundle.tar.gz`` files with sidecars.
        probe_bin (Path): built leanmt-probe binary (pinned commit).
        output_dir (Path): sweep work dir; scorecards land in
            ``output_dir/scorecards/``.
        corpus_dir (Optional[Path]): probe corpora; defaults to the packaged
            traveler-v1 corpora.
        threads (Optional[int]): intra-op thread override for leanmt-probe.
        leanmt_commit (Optional[str]): provenance recorded per scorecard.

    Returns:
        SweepReport: one result per discovered bundle.
    """
    from .config import DEFAULT_CORPUS_DIR

    corpus = corpus_dir or DEFAULT_CORPUS_DIR
    report = SweepReport()
    repack_dir = output_dir / "repacked"
    work_dir = output_dir / "work"
    repack_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    for published in sorted(bundles_dir.glob("*-bundle.tar.gz")):
        result = BundleSweepResult(
            slug=published.name[: -len("-bundle.tar.gz")],
            published_file=published.name,
            published_sha256=sha256_of(published),
        )
        report.results.append(result)
        try:
            repacked = repack_published_bundle(published, work_dir, repack_dir)
            result.repacked_sha256 = sha256_of(repacked)
            result.scorecard = run_bundle_probe(
                repacked,
                probe_bin,
                corpus_dir=corpus,
                output_dir=output_dir / "probe" / result.slug,
                threads=threads,
                leanmt_commit=leanmt_commit,
            )
            result.scorecard_path = (
                output_dir
                / "probe"
                / result.slug
                / "scorecards"
                / f"{result.slug}.json"
            )
        except (SweepError, RunnerError) as error:
            result.error = str(error)
        except Exception as error:  # a swept defect must not kill the sweep
            result.error = f"{type(error).__name__}: {error}"

    # Flatten scorecards into one artifact directory for upload/publish.
    flat = output_dir / "scorecards"
    flat.mkdir(parents=True, exist_ok=True)
    for result in report.results:
        if result.scorecard_path and result.scorecard_path.is_file():
            shutil.copy2(result.scorecard_path, flat / result.scorecard_path.name)
    return report


def build_summary(report: SweepReport) -> dict:
    """
    Builds the machine-readable sweep summary (schema
    ``versta.pack-sweep/1``) consumed by the epic thread and the publish
    gate in NEU-60.
    """
    bundles: List[Dict[str, Any]] = []
    hard_failing = 0
    for result in report.results:
        entry: Dict[str, Any] = {
            "slug": result.slug,
            "published_file": result.published_file,
            "published_sha256": result.published_sha256,
            "repacked_sha256": result.repacked_sha256,
        }
        if result.scorecard is not None:
            entry["acceptance"] = result.scorecard["acceptance"]
            entry["expectation_met"] = result.scorecard["expectation_met"]
            entry["directions"] = [
                {
                    "direction": direction["direction"],
                    "load_ok": direction["load_ok"],
                    **(
                        {
                            "chrf": direction["chrf"],
                            "chrf_floor": direction["chrf_floor"],
                            "direction_check": direction["direction_check"],
                            "digit_preservation": direction["digit_preservation"],
                            "latency_ms": direction["latency_ms"],
                            "rss_peak_kib": direction["rss_peak_kib"],
                        }
                        if direction["load_ok"]
                        else {"load_error": direction["load_error"]}
                    ),
                }
                for direction in result.scorecard["directions"]
            ]
        else:
            entry["acceptance"] = "fail"
            entry["expectation_met"] = False
            entry["error"] = result.error
        if not entry["expectation_met"]:
            hard_failing += 1
        bundles.append(entry)
    return {
        "schema": SWEEP_SUMMARY_SCHEMA,
        "bundles_total": len(bundles),
        "bundles_passing": len(bundles) - hard_failing,
        "bundles_hard_failing": hard_failing,
        "bundles": bundles,
    }


def write_summary(report: SweepReport, output_path: Path) -> None:
    """Writes the sweep summary as stable, diff-friendly JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_summary(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full-catalog pack acceptance sweep: re-pack every published "
        "bundle through the production bundler and probe it (NEU-59)."
    )
    parser.add_argument(
        "--bundles_dir",
        type=Path,
        required=True,
        help="Directory of published <slug>-bundle.tar.gz files (+ .sha256 sidecars).",
    )
    parser.add_argument(
        "--probe_bin", type=Path, required=True, help="Built leanmt-probe binary."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/sweep"),
        help="Sweep work dir; scorecards land in <output_dir>/scorecards/.",
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=None,
        help="Probe corpora; defaults to packaged traveler-v1 corpora.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--leanmt_commit", type=str, default=None, help="Recorded per scorecard."
    )
    args = parser.parse_args()

    report = sweep_bundles(
        args.bundles_dir,
        args.probe_bin,
        args.output_dir,
        corpus_dir=args.corpus_dir,
        threads=args.threads,
        leanmt_commit=args.leanmt_commit,
    )
    summary = build_summary(report)
    write_summary(report, args.output_dir / "sweep-summary.json")

    print(
        f"sweep: {summary['bundles_passing']}/{summary['bundles_total']} bundles "
        f"pass, {summary['bundles_hard_failing']} hard-failing"
    )
    for entry in summary["bundles"]:
        if not entry["expectation_met"]:
            print(f"  FAIL {entry['slug']}: {entry.get('error', 'acceptance fail')}")
    return 0 if summary["bundles_hard_failing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
