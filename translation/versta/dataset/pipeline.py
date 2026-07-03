from collections import defaultdict
from pathlib import Path

import pycountry

from .extractor import (
    create_reversed_shards,
    download_opus_dataset,
    merge_and_dedup,
    smart_sample,
)
from .processor import process_dataset, write_natural_dataset
from .types import LanguagePairConfig


def _download_and_sample(
    source: str,
    target: str,
    entries: list,
    download_dir: Path,
    intermediates_dir: Path,
    seed: int,
) -> list[str]:
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


def run_pipeline(
    config: LanguagePairConfig,
    cache: Path,
    output: Path,
    workers: int,
    seed: int,
    shard_size: int,
    batch_size: int,
) -> None:
    source = config["source"]
    target = config["target"]
    synthetic = config["train"]["synthetic"]
    natural = config["train"]["natural"]

    languages = sorted([source, target])
    output_dir = output / f"{languages[0]}-{languages[1]}"
    download_dir = cache / "corpora"
    intermediates_dir = output_dir / "intermediates"

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

    synthetic_paths: list[str] = []
    natural_paths_by_register: dict[str, list[str]] = defaultdict(list)

    for (corpus, release), plan in corpus_plan.items():
        syn_entries = plan["syn_entries"]
        nat_entries = plan["nat_entries"]
        syn_demand = plan["syn_pairs"]
        nat_demand = plan["nat_pairs"]

        if syn_entries and nat_entries and syn_demand > 0 and nat_demand > 0:
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

        else:
            register = nat_entries[0].get("register", "plain")
            nat_dir = intermediates_dir / "natural" / register
            nat_dir.mkdir(parents=True, exist_ok=True)
            paths = _download_and_sample(
                source, target, nat_entries, download_dir, nat_dir, seed
            )
            natural_paths_by_register[register].extend(paths)

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
