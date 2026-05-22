import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm

from .llm import generate_tonal_translations
from .types import ProcessedEntry


def _md5(text: str) -> str:
    """Computes MD5 hash of the given text.

    Args:
        text (str): Text to hash.

    Returns:
        str: Hexadecimal MD5 hash string.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class ProcessedResult:
    """Container to carry both the processed entries and the original pair data."""

    entries: List[ProcessedEntry]
    prompt: str
    completion: str


def process_dataset(
    input_paths: list[Path],
    output_file: Path,
    intermediates_dir: Path,
    source_lang: str,
    target_lang: str,
    max_workers: int = 4,
    shard_size: int = 10000,
    batch_size: int = 10,
) -> List[ProcessedEntry]:
    """Process multiple input shards with streaming writes and automatic resume.

    Reads each shard file, skips already-processed pairs, and writes results
    incrementally to avoid data loss on failure. Output and checkpoint files
    are sharded at `shard_size` pair boundaries.

    Args:
        input_paths (list[Path]): List of input JSONL shard paths.
        output_file (Path): Base path for the processed JSONL output (stem used for shard naming).
        intermediates_dir (Path): Cache directory for checkpoint files.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        max_workers (int): Number of parallel workers for LLM inference.
        shard_size (int): Number of pairs per shard. Default 10000.
        batch_size (int): Number of pairs to process in a single LLM batch request. Default 10.

    Returns:
        List[ProcessedEntry]: List of ProcessedEntry dicts.
    """
    stem = output_file.stem
    output_parent = output_file.parent
    checkpoint_parent = intermediates_dir

    output_parent.mkdir(parents=True, exist_ok=True)
    checkpoint_parent.mkdir(parents=True, exist_ok=True)

    processed_hashes: set[str] = set()

    # Resume: scan checkpoint shards (primary source for resume state)
    for checkpoint_shard in sorted(checkpoint_parent.glob(f"*.ckpt")):
        print(f"[RESUME] Scanning checkpoint shard: {checkpoint_shard.name}")
        with checkpoint_shard.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_hashes.add(line.strip())

    already_output = len(processed_hashes)
    if already_output > 0:
        print(f"[RESUME] Found {already_output} already-processed pairs")

    # Collect all pairs and filter out already-processed ones
    all_pairs: list[dict] = []
    for shard_path in input_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            all_pairs.extend(json.loads(line) for line in f if line.strip())

    pairs_to_process: list[dict] = []
    skipped = 0
    for pair in all_pairs:
        prompt = pair.get("prompt", "").strip()
        completion = pair.get("completion", "").strip()
        key = f"{prompt}:{completion}"
        pair_hash = _md5(key)

        if pair_hash in processed_hashes:
            skipped += 1
            continue
        pairs_to_process.append(pair)

    total = skipped + len(pairs_to_process)
    if skipped:
        print(f"[RESUME] Skipping {skipped}/{total} already-processed pairs")

    all_entries: List[ProcessedEntry] = []
    file_lock = threading.Lock()
    shards: dict[int, dict] = {}
    current_shard_idx = 0

    def get_shard(idx: int) -> dict:
        return shards.get(idx, {"out": None, "ckpt": None, "pairs": 0})

    def init_shard(idx: int) -> dict:
        output_shard = output_parent / f"{stem}_{idx:05d}.jsonl"
        checkpoint_shard = checkpoint_parent / f"{stem}_{idx:05d}.ckpt"
        f_out = output_shard.open("a", encoding="utf-8")
        f_ckpt = checkpoint_shard.open("a", encoding="utf-8")
        shard = {"out": f_out, "ckpt": f_ckpt, "pairs": 0}
        shards[idx] = shard
        return shard

    def write_results(result: ProcessedResult, shard_idx: int):
        shard = get_shard(shard_idx)
        for entry in result.entries:
            shard["out"].write(json.dumps(entry, ensure_ascii=False) + "\n")
        pair_hash = _md5(f"{result.prompt}:{result.completion}")
        if result.entries:
            shard["ckpt"].write(pair_hash + "\n")

    init_shard(0)

    batches: list[list[dict]] = []
    for i in range(0, len(pairs_to_process), batch_size):
        batches.append(pairs_to_process[i : i + batch_size])

    if batch_size > 1:
        print(f"[BATCH] Processing {len(pairs_to_process)} pairs in {len(batches)} batches (size={batch_size})")

    def process(batch: list[dict]) -> list[ProcessedResult]:
        results = generate_tonal_translations(batch, source_lang, target_lang)
        processed = []
        for pair, translation in zip(batch, results):
            if not translation:
                processed.append(
                    ProcessedResult(entries=[], prompt=pair["prompt"], completion=pair["completion"])
                )
                continue
            entries: List[ProcessedEntry] = []
            for tone, translated in translation.items():
                entry = ProcessedEntry(
                    source=source_lang,
                    target=target_lang,
                    instruction=f"Translate the following text to {target_lang} in a {tone} tone.",
                    input=pair["prompt"],
                    output=translated,
                )
                entries.append(entry)
            processed.append(
                ProcessedResult(entries=entries, prompt=pair["prompt"], completion=pair["completion"])
            )
        return processed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                batch_results = future.result()
                for result in batch_results:
                    all_entries.extend(result.entries)

                    with file_lock:
                        shard = get_shard(current_shard_idx)
                        if shard["out"] is None:
                            for s in list(shards.values()):
                                if s["out"] is not None:
                                    s["out"].close()
                                    s["out"] = None
                                if s["ckpt"] is not None:
                                    s["ckpt"].close()
                                    s["ckpt"] = None
                            current_shard_idx += 1
                            init_shard(current_shard_idx)
                            shard = get_shard(current_shard_idx)

                        write_results(result, current_shard_idx)
                        if result.entries:
                            shard["pairs"] += 1

                        if shard["pairs"] >= shard_size:
                            for s in list(shards.values()):
                                if s["out"] is not None:
                                    s["out"].close()
                                    s["out"] = None
                                if s["ckpt"] is not None:
                                    s["ckpt"].close()
                                    s["ckpt"] = None
                            current_shard_idx += 1
                            init_shard(current_shard_idx)
            except Exception as e:
                print(f"Worker error: {e}")

    return all_entries
