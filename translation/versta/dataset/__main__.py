import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pycountry

from .corpus import filter_corpus_config, load_corpus_config
from .extractor import (
    create_reversed_shards,
    download_opus_dataset,
    merge_and_dedup,
    smart_sample,
)
from .processor import process_dataset, write_natural_dataset
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


def _download_and_sample(
    source: str,
    target: str,
    entries: list,
    download_dir: Path,
    intermediates_dir: Path,
    seed: int,
) -> list[str]:
    """Download and smart-sample a list of corpus entries, returning filtered paths."""
    dataset_paths: list[str] = []
    for entry in entries:
        corpus = entry["corpus"]
        config_pairs = entry["pairs"]
        release = entry.get("release")

        extraction = download_opus_dataset(
            source=source,
            target=target,
            download_dir=download_dir,
            intermediates_dir=intermediates_dir,
            corpus=corpus,
            pairs=config_pairs,
            release=release,
            preprocess=entry.get("preprocess", "raw"),
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

    return dataset_paths


def _merge_shards(
    dataset_paths: list[str], intermediates_dir: Path, shard_size: int
) -> list[Path]:
    """Merge and deduplicate filtered JSONL paths into shard files."""
    shard_files: list[Path] = []
    if len(dataset_paths) > 1:
        shard_path = intermediates_dir / "merged"
        merge_and_dedup(
            filtered_paths=dataset_paths,
            filtered_file_path=shard_path,
            shard_size=shard_size,
        )
        for shard in sorted(shard_path.parent.glob(shard_path.stem + "_*.jsonl")):
            shard_files.append(shard)
    elif len(dataset_paths) == 1:
        shard_files.append(Path(dataset_paths[0]))
    return shard_files


def _split_sampled_file(
    source_path: Path,
    syn_path: Path,
    nat_path: Path,
    syn_count: int,
) -> None:
    """Split a sampled JSONL file into synthetic and natural portions.

    The first `syn_count` lines go to the synthetic file; the remainder
    go to the natural file.  This preserves the deterministic ordering
    produced by ``smart_sample``, so the two subsets are non-overlapping.
    """
    with open(source_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    with (
        open(syn_path, "w", encoding="utf-8") as f_syn,
        open(nat_path, "w", encoding="utf-8") as f_nat,
    ):
        for i, line in enumerate(lines):
            if i < syn_count:
                f_syn.write(line)
            else:
                f_nat.write(line)


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
        synthetic = config["train"]["synthetic"]
        natural = config["train"]["natural"]

        languages = sorted([source, target])
        output_dir = output / f"{languages[0]}-{languages[1]}"
        download_dir = cache / "corpora"
        intermediates_dir = output_dir / "intermediates"

        cache.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        # Phase 1 — Group all corpora and determine overlap
        # ---------------------------------------------------------------
        corpus_plan: dict[tuple[str, str | None], dict] = defaultdict(
            lambda: {
                "syn_entries": [],
                "nat_entries": [],
                "syn_pairs": 0,
                "nat_pairs": 0,
            }
        )
        for entry in synthetic:
            key = (entry["corpus"], entry.get("release"))
            p = corpus_plan[key]
            p["syn_entries"].append(entry)
            p["syn_pairs"] += entry.get("pairs", 0) or 0

        for entry in natural:
            key = (entry["corpus"], entry.get("release"))
            p = corpus_plan[key]
            p["nat_entries"].append(entry)
            p["nat_pairs"] += entry.get("pairs", 0) or 0

        # ---------------------------------------------------------------
        # Phase 2 — Download, sample, and route to the correct path
        # ---------------------------------------------------------------
        synthetic_paths: list[str] = []
        natural_paths_by_register: dict[str, list[str]] = defaultdict(list)

        for (corpus, release), plan in corpus_plan.items():
            syn_entries = plan["syn_entries"]
            nat_entries = plan["nat_entries"]
            syn_demand = plan["syn_pairs"]
            nat_demand = plan["nat_pairs"]

            if syn_entries and nat_entries and syn_demand > 0 and nat_demand > 0:
                # Both arrays reference this corpus → download once, sample
                # the combined total, then split deterministically.
                coord_dir = intermediates_dir / "coordinated"
                coord_dir.mkdir(parents=True, exist_ok=True)

                preprocess = syn_entries[0].get("preprocess")
                if preprocess is None:
                    preprocess = "raw"

                total_demand = syn_demand + nat_demand
                extraction = download_opus_dataset(
                    source=source,
                    target=target,
                    download_dir=download_dir,
                    intermediates_dir=coord_dir,
                    corpus=corpus,
                    pairs=total_demand,
                    release=release,
                    preprocess=preprocess,
                )

                sampled_path = coord_dir / f"{corpus}_{source}-{target}.sampled.jsonl"
                smart_sample(
                    jsonl_path=extraction["output_file"],
                    output_path=sampled_path,
                    pairs=total_demand,
                    seed=seed,
                )

                syn_file = coord_dir / f"{corpus}_{source}-{target}.synthetic.jsonl"
                nat_file = coord_dir / f"{corpus}_{source}-{target}.natural.jsonl"
                _split_sampled_file(sampled_path, syn_file, nat_file, syn_demand)

                synthetic_paths.append(str(syn_file))
                register = nat_entries[0].get("register", "plain")
                natural_paths_by_register[register].append(str(nat_file))

            elif syn_entries:
                syn_dir = intermediates_dir / "synthetic"
                syn_dir.mkdir(parents=True, exist_ok=True)
                paths = _download_and_sample(
                    source, target, syn_entries, download_dir, syn_dir, seed
                )
                synthetic_paths.extend(paths)

            else:  # nat_entries only
                register = nat_entries[0].get("register", "plain")
                nat_dir = intermediates_dir / "natural" / register
                nat_dir.mkdir(parents=True, exist_ok=True)
                # De-duplicate downloads when multiple nat entries share the
                # same corpus (unusual, but handled by _download_and_sample).
                paths = _download_and_sample(
                    source, target, nat_entries, download_dir, nat_dir, seed
                )
                natural_paths_by_register[register].extend(paths)

        # ---------------------------------------------------------------
        # Phase 3a — Synthetic path: merge → LLM → reversed → LLM
        # ---------------------------------------------------------------
        if synthetic_paths:
            syn_merge_dir = intermediates_dir / "synthetic_merged"
            syn_merge_dir.mkdir(parents=True, exist_ok=True)
            shard_files = _merge_shards(synthetic_paths, syn_merge_dir, shard_size)

            if shard_files:
                output_dir.mkdir(parents=True, exist_ok=True)

                forward_checkpoints = syn_merge_dir / f"{source}-{target}"
                forward_checkpoints.mkdir(parents=True, exist_ok=True)
                process_dataset(
                    input_paths=shard_files,
                    intermediates_dir=forward_checkpoints,
                    output_file=output_dir / "dataset.jsonl",
                    source_lang=source,
                    target_lang=target,
                    max_workers=workers,
                    shard_size=shard_size,
                    batch_size=batch_size,
                )

                reversed_shards = create_reversed_shards(shard_files, syn_merge_dir)
                reverse_checkpoints = syn_merge_dir / f"{target}-{source}"
                reverse_checkpoints.mkdir(parents=True, exist_ok=True)
                process_dataset(
                    input_paths=reversed_shards,
                    intermediates_dir=reverse_checkpoints,
                    output_file=output_dir / "dataset.jsonl",
                    source_lang=target,
                    target_lang=source,
                    max_workers=workers,
                    shard_size=shard_size,
                    batch_size=batch_size,
                )

        # ---------------------------------------------------------------
        # Phase 3b — Natural path: merge → direct-write → reversed → direct-write
        # ---------------------------------------------------------------
        if natural_paths_by_register:
            target_lang_obj = pycountry.languages.get(alpha_2=target)
            target_name = target_lang_obj.name if target_lang_obj else target
            source_lang_obj = pycountry.languages.get(alpha_2=source)
            source_name = source_lang_obj.name if source_lang_obj else source

            register_instructions = {
                "plain": f"Translate to {target_name}.",
                "neutral": f"Translate to neutral {target_name}.",
                "formal": f"Translate to formal {target_name}.",
                "casual": f"Translate to casual {target_name}.",
            }

            for register, paths in natural_paths_by_register.items():
                nat_reg_dir = intermediates_dir / "natural" / register
                nat_reg_dir.mkdir(parents=True, exist_ok=True)
                shard_files = _merge_shards(paths, nat_reg_dir, shard_size)

                if shard_files:
                    output_dir.mkdir(parents=True, exist_ok=True)

                    instruction = register_instructions.get(
                        register, register_instructions["plain"]
                    )
                    forward_checkpoints = nat_reg_dir / f"{source}-{target}"
                    forward_checkpoints.mkdir(parents=True, exist_ok=True)
                    write_natural_dataset(
                        input_paths=shard_files,
                        intermediates_dir=forward_checkpoints,
                        output_file=output_dir / "dataset.jsonl",
                        source_lang=source,
                        target_lang=target,
                        instruction=instruction,
                        shard_size=shard_size,
                    )

                    reversed_shards = create_reversed_shards(shard_files, nat_reg_dir)
                    reverse_checkpoints = nat_reg_dir / f"{target}-{source}"
                    reverse_checkpoints.mkdir(parents=True, exist_ok=True)
                    write_natural_dataset(
                        input_paths=reversed_shards,
                        intermediates_dir=reverse_checkpoints,
                        output_file=output_dir / "dataset.jsonl",
                        source_lang=target,
                        target_lang=source,
                        instruction=f"Translate to {register} {source_name}."
                        if register != "plain"
                        else f"Translate to {source_name}.",
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
