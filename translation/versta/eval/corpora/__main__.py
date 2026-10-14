"""Fetch, verify or build the pinned evaluation corpus snapshots.

Usage:
    uv run python -m versta.eval.corpora fetch  [--manifest ...] [--output_dir ...] [--check]
    uv run python -m versta.eval.corpora verify --corpus_dir DIR [--manifest ...]
    uv run python -m versta.eval.corpora build  [--revision N] [--upstream_tarball P] ...
"""

import os

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .config import FLORES200_UPSTREAM
from .manifest import load_manifest
from .pipeline import (
    build_snapshot,
    fetch_snapshot,
    fetch_upstream,
    verify_corpus,
)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Manage the hash-pinned evaluation corpus snapshots
        (see evals/corpora/README.md). 'fetch' reproduces the corpus recorded
        in the checked-in manifest byte-identically; 'build' creates a new
        snapshot revision from the pinned upstream dataset and updates the
        manifest.""",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser(
        "fetch", help="Download and verify the corpus recorded in the manifest."
    )
    fetch.add_argument(
        "--manifest",
        type=Path,
        default=Path("evals/corpora/flores200-dev.json"),
        help="Path of the checked-in corpus manifest.",
    )
    fetch.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/corpora"),
        help="Cache directory the corpus is extracted into.",
    )
    fetch.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Only verify an already-fetched corpus; never download.",
    )

    verify = commands.add_parser(
        "verify", help="Verify an extracted corpus against the manifest."
    )
    verify.add_argument(
        "--manifest",
        type=Path,
        default=Path("evals/corpora/flores200-dev.json"),
        help="Path of the checked-in corpus manifest.",
    )
    verify.add_argument(
        "--corpus_dir",
        type=Path,
        required=True,
        help="Root of the extracted corpus (contains the dev/ directory).",
    )

    build = commands.add_parser(
        "build", help="Build a new snapshot revision from the upstream dataset."
    )
    build.add_argument(
        "--revision",
        type=int,
        default=1,
        help="Snapshot revision; bump for any content change.",
    )
    build.add_argument(
        "--catalog",
        type=Path,
        default=Path("models.json"),
        help="The checked-in model catalog to derive directions from.",
    )
    build.add_argument(
        "--upstream_tarball",
        type=Path,
        default=None,
        help="Local copy of the upstream archive; downloaded from "
        "--upstream_url when omitted.",
    )
    build.add_argument(
        "--upstream_url",
        type=str,
        default=FLORES200_UPSTREAM.url,
        help="URL of the upstream dataset archive.",
    )
    build.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/corpora"),
        help="Where the snapshot artifact and checksum are written.",
    )
    build.add_argument(
        "--manifest",
        type=Path,
        default=Path("evals/corpora/flores200-dev.json"),
        help="Where the checked-in manifest is (re)written.",
    )

    return parser.parse_args()


def main() -> None:
    """Dispatches the selected corpora subcommand."""
    args = parse_args()

    if args.command == "fetch":
        corpus_dir = fetch_snapshot(
            manifest_path=args.manifest, output_dir=args.output_dir, check=args.check
        )
        print(f"Corpus ready: {corpus_dir}")
    elif args.command == "verify":
        verify_corpus(args.corpus_dir, load_manifest(args.manifest))
        print(f"Corpus verified: {args.corpus_dir}")
    else:
        upstream_tarball = args.upstream_tarball or fetch_upstream(
            args.upstream_url, args.output_dir
        )
        build_snapshot(
            upstream_tarball=upstream_tarball,
            catalog_path=args.catalog,
            revision=args.revision,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )


if __name__ == "__main__":
    main()
