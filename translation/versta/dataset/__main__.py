import os
from argparse import ArgumentParser
from pathlib import Path

from .corpus import filter_corpus_config, load_corpus_config
from .extractor import (
    download_opus_dataset,
    merge_and_dedup,
    smart_sample,
)
from .processor import process_dataset
from .utils import remove_folder


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="Download OPUS corpus(ata) and process for tonal translations.",
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default="corpora.json",
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
        default=Path("output/dataset"),
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
        "--pairs",
        type=int,
        default=0,
        help="Maximum number of sentence pairs to extract from OPUS.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for LLM inference.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
        "This will default to False if not specified.",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of pairs per shard for input merging and output processing.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of sentence pairs to process in a single LLM batch request.",
    )

    parsed_args = parser.parse_args()

    if (parsed_args.source is not None) != (parsed_args.target is not None):
        parser.error("--source and --target must both be provided together, or neither")

    return parsed_args


def main(
    corpus_config: list,
    cache: Path = Path("cache"),
    output: Path = Path("output"),
    pairs: int | None = None,
    workers: int = 4,
    seed: int = 42,
    keep_intermediates: bool = False,
    shard_size: int = 10000,
    batch_size: int = 20,
) -> None:
    """Download corpus(ata), filter, deduplicate, and process for tonal translations.

    Args:
        corpus_config: List of LanguagePairConfig dicts.
    """
    for config in corpus_config:
        source = config["source"]
        target = config["target"]
        corpora = config["corpora"]

        output_dir = output / f"{source}-{target}"
        download_dir = cache / "corpora"
        intermediates_dir = output / f"{source}-{target}" / "intermediates"

        output_dir.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        dataset_paths = []
        for corpus_config_entry in corpora:
            corpus = corpus_config_entry["corpus"]
            config_pairs = corpus_config_entry["pairs"]
            release = corpus_config_entry["release"]

            extraction = download_opus_dataset(
                source=source,
                target=target,
                download_dir=download_dir,
                intermediates_dir=intermediates_dir,
                corpus=corpus,
                pairs=config_pairs,
                release=release,
            )

            raw_jsonl_path = extraction["output_file"]
            filtered_jsonl_path = (
                intermediates_dir / f"{corpus}_{source}-{target}.filtered.jsonl"
            )

            smart_sample(
                jsonl_path=raw_jsonl_path,
                output_path=filtered_jsonl_path,
                pairs=config_pairs,
                seed=seed,
            )

            dataset_paths.append(str(filtered_jsonl_path))

        shard_files: list[Path] = []
        if len(dataset_paths) > 1:
            shard_path = intermediates_dir / "all_filtered_merged"
            merge_and_dedup(
                filtered_paths=dataset_paths,
                filtered_file_path=shard_path,
                shard_size=shard_size,
            )

            for shard in sorted(shard_path.parent.glob(shard_path.name + "_*.jsonl")):
                shard_files.append(shard)
        else:
            shard_files.append(Path(dataset_paths[0]))

        print(f"Processing the following shards: {shard_files}")

        process_dataset(
            input_paths=shard_files,
            intermediates_dir=intermediates_dir,
            output_file=output_dir / "dataset.jsonl",
            source_lang=source,
            target_lang=target,
            max_workers=workers,
            shard_size=shard_size,
            batch_size=batch_size,
        )

        if not keep_intermediates:
            remove_folder(intermediates_dir)
            print("Intermediates files cleaned.")


if __name__ == "__main__":
    args = parse_args()

    if args.pairs == 0:
        args.pairs = None

    corpus_configs = load_corpus_config(args.corpus)
    corpus_configs = filter_corpus_config(corpus_configs, args.source, args.target)

    main(
        corpus_config=corpus_configs,
        cache=args.cache,
        output=args.output,
        pairs=args.pairs,
        workers=args.workers,
        seed=args.seed,
        keep_intermediates=args.keep_intermediates,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
    )
