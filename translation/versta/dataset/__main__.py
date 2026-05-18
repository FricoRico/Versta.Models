import json
import os
from argparse import ArgumentParser
from pathlib import Path

from .extractor import download_opus_dataset, merge_and_dedup, smart_sample
from .processor import process_dataset
from .types import CorpusConfig, LanguagePairConfig
from .utils import remove_folder


def _create_reversed_shards(
    input_shards: list[Path], intermediates_dir: Path
) -> list[Path]:
    """Swap prompt/completion in each shard, write reversed shards to disk."""
    reversed_paths = []
    for i, shard in enumerate(input_shards):
        reversed_path = intermediates_dir / f"mirrored_{i:05d}.jsonl"
        with (
            shard.open("r", encoding="utf-8") as fin,
            reversed_path.open("w", encoding="utf-8") as fout,
        ):
            for line in fin:
                pair = json.loads(line)
                pair["prompt"], pair["completion"] = pair["completion"], pair["prompt"]
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
        reversed_paths.append(reversed_path)
    return reversed_paths


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
        "--output",
        type=Path,
        default=Path("output"),
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
        default=6,
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

    return parser.parse_args()


def load_corpus_config(corpus_arg: str) -> list[LanguagePairConfig]:
    """Load corpus configuration from a JSON file."""
    corpus_path = Path(corpus_arg)

    if not corpus_path.exists() or corpus_path.suffix != ".json":
        raise ValueError(
            f"'{corpus_arg}' is not a valid JSON file. "
            "Please provide a path to a corpora JSON config file."
        )

    with open(corpus_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    typed_configs = []
    for config in configs:
        corpora = []
        for c in config.get("corpora", []):
            corpora.append(
                CorpusConfig(
                    corpus=c["corpus"],
                    pairs=c.get("pairs"),
                    release=c.get("release"),
                )
            )

        typed_configs.append(
            LanguagePairConfig(
                source=config["source"],
                target=config["target"],
                corpora=corpora,
            )
        )

    return typed_configs


def main(
    corpus_config: list[LanguagePairConfig],
    cache: Path,
    output: Path,
    pairs: int | None = None,
    workers: int = 4,
    seed: int = 42,
    keep_intermediates: bool = False,
    shard_size: int = 10000,
) -> None:
    """Download corpus(ata), filter, deduplicate, and process for tonal translations.

    Args:
        corpus_config: List of LanguagePairConfig dicts.
    """
    for config in corpus_config:
        source = config["source"]
        target = config["target"]
        corpora = config["corpora"]

        print(f"\n{'='*60}")
        print(f"Processing language pair: {source} -> {target}")
        print(f"{'='*60}\n")

        output_dir = output / f"{source}-{target}"
        cache_dir = cache / f"{source}-{target}"
        download_dir = cache_dir / "corpora"
        intermediates_dir = cache_dir / "intermediates"

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
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
                intermediates_dir=filtered_jsonl_path,
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
        )

        reversed_shards = _create_reversed_shards(shard_files, intermediates_dir)
        print(f"Mirrored shards: {len(reversed_shards)}")
        process_dataset(
            input_paths=reversed_shards,
            intermediates_dir=intermediates_dir,
            output_file=output_dir / "dataset.jsonl",
            source_lang=target,
            target_lang=source,
            max_workers=workers,
            shard_size=shard_size,
        )

        if not keep_intermediates:
            remove_folder(intermediates_dir)
            print("Intermediates files cleaned.")


if __name__ == "__main__":
    args = parse_args()

    if args.pairs == 0:
        args.pairs = None

    corpus_configs = load_corpus_config(args.corpus)

    main(
        corpus_config=corpus_configs,
        cache=args.cache,
        output=args.output,
        pairs=args.pairs,
        workers=args.workers,
        seed=args.seed,
        keep_intermediates=args.keep_intermediates,
        shard_size=args.shard_size,
    )
