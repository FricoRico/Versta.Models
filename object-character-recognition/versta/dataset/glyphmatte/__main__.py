"""Materialize the synthetic glyphmatte strip dataset to parquet shards.

Syncs the pinned fonts/word lists on first use, then writes HF-convention
shards + metadata.jsonl. Defaults compose with the training pipeline, which
looks for exactly this output location.

CLI: uv run python -m versta.dataset.glyphmatte [--output_dir output/dataset/glyphmatte]
"""

import argparse

from pathlib import Path

from .config import DATASET
from .materialize import materialize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/dataset/glyphmatte"),
        help="Output directory for the parquet shards + metadata.jsonl.",
    )
    parser.add_argument("--n", type=int, default=DATASET.n, help="Train strip count.")
    parser.add_argument(
        "--shard_size", type=int, default=DATASET.shard_size, help="Strips per shard."
    )
    parser.add_argument("--seed", type=int, default=DATASET.seed, help="Dataset seed.")
    parser.add_argument(
        "--val_n", type=int, default=DATASET.val_n, help="Validation strip count."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_dataset(args.output_dir, args.n, args.val_n, args.shard_size, args.seed)


if __name__ == "__main__":
    main()
