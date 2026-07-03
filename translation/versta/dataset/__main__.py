import os
from argparse import ArgumentParser
from pathlib import Path

from .corpus import filter_corpus_config, load_corpus_config
from .pipeline import run_pipeline
from .utils import remove_folder


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="Download OPUS corpus(ata) and process for tonal translations.",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpora.json"),
        help="Path to corpora JSON config file.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source language code to filter to a single language pair.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target language code to filter to a single language pair.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dataset/train"),
        help="Path to the output folder.",
        action="store",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache"),
        help="Path to the cache folder.",
        action="store",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1778419142,
        help="Random seed for deterministic sampling.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for LLM inference.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of sentence pairs to process in a single LLM batch request.",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of pairs per shard for input merging and output processing.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
        "This will default to False if not specified.",
    )

    parsed_args = parser.parse_args()

    if (parsed_args.source is not None) != (parsed_args.target is not None):
        parser.error("--source and --target must both be provided together, or neither")

    return parsed_args


def main(
    corpus_config: list,
    cache: Path,
    output: Path,
    workers: int,
    seed: int,
    shard_size: int,
    batch_size: int,
    keep_intermediates: bool = False,
) -> None:
    """Download corpus(ata), filter, deduplicate, and process for tonal translations.

    Args:
        corpus_config: List of LanguagePairConfig dicts.
    """
    for config in corpus_config:
        source = config["source"]
        target = config["target"]

        languages = sorted([source, target])

        output_dir = output / f"{languages[0]}-{languages[1]}"

        download_dir = cache / "corpora"
        intermediates_dir = output_dir / "intermediates"

        output_dir.mkdir(parents=True, exist_ok=True)
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        cache.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        run_pipeline(
            config=config,
            cache=cache,
            output=output,
            workers=workers,
            seed=seed,
            shard_size=shard_size,
            batch_size=batch_size,
        )

        if not keep_intermediates:
            remove_folder(intermediates_dir)
            print("Intermediates files cleaned.")


if __name__ == "__main__":
    args = parse_args()

    corpus_configs = load_corpus_config(args.corpus)
    corpus_configs = filter_corpus_config(corpus_configs, args.source, args.target)

    main(
        corpus_config=corpus_configs,
        cache=args.cache,
        output=args.output,
        workers=args.workers,
        seed=args.seed,
        keep_intermediates=args.keep_intermediates,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
    )
