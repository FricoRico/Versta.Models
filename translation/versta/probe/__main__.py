import argparse

from pathlib import Path

from .config import DEFAULT_CORPUS_DIR
from .install import InstallError
from .runner import RunnerError, run_bundle_probe, summarize
from .scorecard import EXPECT_DIRECTION_FAIL, EXPECT_PASS
from ..schema import ValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack acceptance probe runner (NEU-60): installs a translation "
            "bundle through the app's metadata/parse path, loads every "
            "declared direction via the pinned leanmt-probe binary, "
            "translates the traveler probe corpus, and writes a scorecard "
            "with load / chrF / digit-preservation / latency / RSS verdicts. "
            "Exit 0 iff the outcome matches --expect."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Bundle tarball to probe (.tar.gz produced by versta.bundle).",
    )
    parser.add_argument(
        "--probe_bin",
        type=Path,
        required=True,
        help="Path of the built leanmt-probe binary (Versta.Leanmt, pinned commit).",
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory with <src>-<tgt>.json probe corpora; defaults to the packaged traveler-v1 corpora.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/probe"),
        help="Working + scorecard output directory.",
    )
    parser.add_argument(
        "--expect",
        choices=[EXPECT_PASS, EXPECT_DIRECTION_FAIL],
        default=EXPECT_PASS,
        help="Expected outcome: 'pass' for real packs, 'direction-fail' for the adversarial control pack.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Scorecard filename stem override (defaults to the bundle slug, plus '.adversarial' for direction-fail runs).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Intra-op thread override for leanmt-probe; default keeps leanmt's process default.",
    )
    parser.add_argument(
        "--leanmt_commit",
        type=str,
        default=None,
        help="Pinned Versta.Leanmt commit the probe binary was built from (provenance in the scorecard).",
    )
    return parser.parse_args()


def main() -> int:
    """
    Importable entry point. Returns the process exit code: 0 outcome matches
    expectation, 1 unexpected acceptance outcome, 2 harness/setup failure.
    """
    args = parse_args()
    try:
        scorecard = run_bundle_probe(
            bundle_path=args.bundle,
            probe_bin=args.probe_bin,
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            threads=args.threads,
            expectation=args.expect,
            label=args.label,
            leanmt_commit=args.leanmt_commit,
        )
    except (RunnerError, InstallError, ValidationError, OSError) as error:
        print(f"probe run failed: {error}")
        return 2

    print(summarize(scorecard))
    print(f"scorecard written under {args.output_dir / 'scorecards'}")
    return 0 if scorecard["expectation_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
