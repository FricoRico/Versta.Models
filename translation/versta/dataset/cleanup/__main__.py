import os
from argparse import ArgumentParser
from pathlib import Path

from .cleanup import cleanup_all, cleanup_lang_pair


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="Remove stale entries from processed dataset output and checkpoints.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source language code. Must be used with --target.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target language code. Must be used with --source.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dataset"),
        help="Base output directory containing language pair folders.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview changes without modifying anything.",
    )

    parsed_args = parser.parse_args()

    if (parsed_args.source is not None) != (parsed_args.target is not None):
        parser.error("--source and --target must both be provided together, or neither")

    return parsed_args


def main(
    source: str | None,
    target: str | None,
    output: Path = Path("output/dataset"),
    dry_run: bool = False,
) -> None:
    if source and target:
        langs = sorted([source, target])
        pair_dir = output / f"{langs[0]}-{langs[1]}"
        if not pair_dir.exists():
            print(f"Error: Language pair directory not found: {pair_dir}")
            return

        cleanup_lang_pair(pair_dir, dry_run=dry_run)
    else:
        cleanup_all(output, dry_run=dry_run)


if __name__ == "__main__":
    args = parse_args()
    main(
        source=args.source,
        target=args.target,
        output=args.output,
        dry_run=args.dry_run,
    )
