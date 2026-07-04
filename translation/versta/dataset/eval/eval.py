import hashlib
import json
from pathlib import Path

import pycountry

from ..extractor import download_opus_dataset, merge_and_dedup, smart_sample
from ..processor import process_dataset
from ..types import CorpusConfig, ProcessedEntry


def _load_training_hashes(
    dataset_name: str,
    split: str = "train",
    max_rows: int = 100000,
) -> set[str]:
    try:
        from datasets import load_dataset as loader

        dataset = loader(dataset_name, split=split, streaming=True).take(max_rows)
    except Exception as e:
        print(
            f"Warning: could not load '{dataset_name}' for overlap "
            f"filtering — skipping dedup against training set: {e}"
        )
        return set()

    hashes: set[str] = set()
    for i, row in enumerate(dataset):
        src = row.get("input", "")
        if src:
            hashes.add(hashlib.md5(src.encode("utf-8")).hexdigest())
    return hashes


def _language_name(code: str) -> str:
    """Resolve an ISO 639-1 language code to its full English name.

    Args:
        code: Two-letter ISO language code (e.g. ``"nl"``).

    Returns:
        Full language name (e.g. ``"Dutch"``), or the code itself if unknown.
    """
    lang = pycountry.languages.get(alpha_2=code)
    return lang.name if lang else code


def _download_sample_merge(
    configs: list[CorpusConfig],
    source: str,
    target: str,
    download_dir: Path,
    intermediates_dir: Path,
    seed: int,
    training_hashes: set[str],
) -> list[dict]:
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths: list[str] = []
    total_pairs = 0
    for entry in configs:
        corpus = entry["corpus"]
        config_pairs = entry.get("pairs")
        if config_pairs is not None:
            total_pairs += config_pairs

        extraction = download_opus_dataset(
            source=source,
            target=target,
            download_dir=download_dir,
            intermediates_dir=intermediates_dir,
            corpus=corpus,
            pairs=config_pairs,
            release=entry.get("release"),
            preprocess=entry.get("preprocess", "raw"),
            skip_hashes=training_hashes if training_hashes else None,
        )

        filtered_path = intermediates_dir / f"{corpus}_{source}-{target}.filtered.jsonl"
        smart_sample(
            jsonl_path=extraction["output_file"],
            output_path=filtered_path,
            pairs=config_pairs,
            seed=seed,
            sample_mode="tail",
        )
        dataset_paths.append(str(filtered_path))

    if not dataset_paths:
        return []

    merged_path = intermediates_dir / "merged"
    merge_and_dedup(
        filtered_paths=dataset_paths,
        filtered_file_path=merged_path,
        shard_size=max(10000, total_pairs * 2),
    )

    pairs: list[dict] = []
    for shard in sorted(merged_path.parent.glob(merged_path.stem + "_*.jsonl")):
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    return pairs


def _format_natural_entries(
    pairs: list[dict],
    instruction: str,
    source: str,
    target: str,
) -> list[ProcessedEntry]:
    """Format parallel pairs directly into ProcessedEntry rows."""
    return [
        ProcessedEntry(
            source=source,
            target=target,
            instruction=instruction,
            input=pair["prompt"],
            output=pair["completion"],
            method="natural",
        )
        for pair in pairs
    ]


def generate_eval_dataset(
    source: str,
    target: str,
    dataset: str,
    synthetic_configs: list[CorpusConfig],
    natural_configs: list[CorpusConfig],
    split: str = "train",
    seed: int = 1778419142,
    download_dir: Path = Path("cache/corpora"),
    intermediates_dir: Path = Path("output/eval/intermediates"),
    batch_size: int = 20,
    max_workers: int = 4,
    output_path: Path = Path("eval_data.jsonl"),
) -> list[ProcessedEntry]:
    """Generate a dedicated evaluation dataset with synthetic and natural entries.

    Pipeline:
        1. Load training-set source hashes for overlap filtering.
        2. Download OPUS data, skipping any source that appears in training.
        3. Quality-filter and merge across corpora.
        4. For synthetic pairs: send through ``process_dataset`` for parallel
           LLM tonal expansion (3 tones per pair).
        5. For natural pairs: write directly as ``ProcessedEntry`` rows.
        6. Combine both, write to JSONL.

    Args:
        source: Source ISO language code.
        target: Target ISO language code.
        dataset: HuggingFace dataset name for the existing training set.
        eval_synthetic_configs: List of ``CorpusConfig`` for LLM-expanded eval data.
        eval_natural_configs: List of ``CorpusConfig`` for direct-write eval data.
        train_split: Split name of the existing training set.
        seed: Random seed for deterministic sampling.
        download_dir: Directory for cached OPUS downloads.
        intermediates_dir: Directory for intermediate files.
        batch_size: Number of sentence pairs per LLM batch request.
        max_workers: Number of parallel workers for LLM inference.
        output_path: Path to write the final JSONL file.

    Returns:
        List of generated ``ProcessedEntry`` dicts.
    """
    source_lang_name = _language_name(source)
    target_lang_name = _language_name(target)
    training_hashes = _load_training_hashes(dataset, split)

    all_entries: list[ProcessedEntry] = []

    # ------------------------------------------------------------------
    # Synthetic eval path — LLM tonal expansion via process_dataset
    # ------------------------------------------------------------------
    if synthetic_configs:
        print("Processing eval synthetic corpora ...")
        pairs = _download_sample_merge(
            synthetic_configs,
            source,
            target,
            download_dir,
            intermediates_dir / "synthetic",
            seed,
            training_hashes,
        )
        if pairs:
            syn_dir = intermediates_dir / "synthetic"
            temp_input = syn_dir / "input.jsonl"
            with open(temp_input, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            syn_entries = process_dataset(
                input_paths=[temp_input],
                output_file=syn_dir / "results.jsonl",
                intermediates_dir=syn_dir / "llm",
                source_lang=source,
                target_lang=target,
                max_workers=max_workers,
                batch_size=batch_size,
            )
            for entry in syn_entries:
                entry["method"] = "synthetic"
            all_entries.extend(syn_entries)

            rev_input = syn_dir / "input_rev.jsonl"
            with open(rev_input, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(
                        json.dumps(
                            {"prompt": p["completion"], "completion": p["prompt"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            rev_entries = process_dataset(
                input_paths=[rev_input],
                output_file=syn_dir / "results_rev.jsonl",
                intermediates_dir=syn_dir / "llm_rev",
                source_lang=target,
                target_lang=source,
                max_workers=max_workers,
                batch_size=batch_size,
            )
            for entry in rev_entries:
                entry["method"] = "synthetic"
            all_entries.extend(rev_entries)

            print(f"  → synthetic: {len(syn_entries)} fwd + {len(rev_entries)} rev")
        else:
            print("  (no synthetic pairs available)")

    # ------------------------------------------------------------------
    # Natural eval path — direct write
    # ------------------------------------------------------------------
    if natural_configs:
        print("Processing eval natural corpora ...")

        pairs = _download_sample_merge(
            natural_configs,
            source,
            target,
            download_dir,
            intermediates_dir / "natural",
            seed,
            training_hashes,
        )
        if pairs:
            nat_entries = _format_natural_entries(
                pairs, f"Translate to {target_lang_name}.", source, target
            )
            all_entries.extend(nat_entries)

            rev_pairs = [
                {"prompt": p["completion"], "completion": p["prompt"]} for p in pairs
            ]
            rev_entries = _format_natural_entries(
                rev_pairs, f"Translate to {source_lang_name}.", target, source
            )
            all_entries.extend(rev_entries)

            print(f"  → natural: {len(nat_entries)} fwd + {len(rev_entries)} rev")
        else:
            print("  (no natural pairs available)")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    if not all_entries:
        print("Warning: no eval entries were produced.")
        return []

    with output_path.open("w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_entries)} total eval entries to {output_path}")
    return all_entries
