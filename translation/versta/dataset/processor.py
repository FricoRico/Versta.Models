import hashlib
import json
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pycountry
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

    all_entries: List[ProcessedEntry] = []
    file_lock = threading.Lock()
    shards: dict[int, dict] = {}
    current_shard_idx = 0
    skipped = 0

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

    def process(batch: list[dict]) -> list[ProcessedResult]:
        results = generate_tonal_translations(batch, source_lang, target_lang)
        processed = []
        for pair, translation in zip(batch, results):
            if not translation:
                processed.append(
                    ProcessedResult(entries=[], prompt=pair["prompt"], completion=pair["completion"])
                )
                continue
            target_lang_obj = pycountry.languages.get(alpha_2=target_lang)
            target_name = target_lang_obj.name if target_lang_obj else target_lang

            entries: List[ProcessedEntry] = []
            for tone, translated in translation.items():
                entry = ProcessedEntry(
                    source=source_lang,
                    target=target_lang,
                    instruction=f"Translate to {tone.lower()} {target_name}.",
                    input=pair["prompt"],
                    output=translated,
                    method="synthetic",
                )
                entries.append(entry)
            processed.append(
                ProcessedResult(entries=entries, prompt=pair["prompt"], completion=pair["completion"])
            )
        return processed

    # Count batches for progress bar (fast local pass)
    total_batches = 0
    for shard_path in input_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            acc = 0
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                prompt = pair.get("prompt", "").strip()
                completion = pair.get("completion", "").strip()
                key = f"{prompt}:{completion}"
                if _md5(key) in processed_hashes:
                    continue
                acc += 1
                if acc >= batch_size:
                    total_batches += 1
                    acc = 0
            if acc > 0:
                total_batches += 1

    if total_batches > 0:
        print(f"[BATCH] Processing ~{total_batches} batches (batch_size={batch_size})")

    # Stream input pairs, skipping already-processed ones, yielding batches
    def batch_generator():
        nonlocal skipped
        batch = []
        for shard_path in input_paths:
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    pair = json.loads(line)
                    prompt = pair.get("prompt", "").strip()
                    completion = pair.get("completion", "").strip()
                    key = f"{prompt}:{completion}"
                    pair_hash = _md5(key)

                    if pair_hash in processed_hashes:
                        skipped += 1
                        continue

                    batch.append(pair)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    def handle_result(result: ProcessedResult):
        nonlocal current_shard_idx
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: set[Future] = set()
        batch_iter = iter(batch_generator())

        # Prime with initial batches (bounded in-flight work)
        for _ in range(max_workers * 2):
            try:
                pending.add(executor.submit(process, next(batch_iter)))
            except StopIteration:
                break

        pbar = tqdm(total=total_batches, desc="Processing", unit="batch")

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    for result in future.result():
                        handle_result(result)
                except Exception as e:
                    print(f"Worker error: {e}")
                pbar.update()

            for _ in range(len(done)):
                try:
                    pending.add(executor.submit(process, next(batch_iter)))
                except StopIteration:
                    break

        pbar.close()

    if skipped:
        print(f"[RESUME] Skipped {skipped} already-processed pairs")

    return all_entries


def write_natural_dataset(
    input_paths: list[Path],
    output_file: Path,
    intermediates_dir: Path,
    source_lang: str,
    target_lang: str,
    instruction: str,
    shard_size: int = 10000,
    start_shard: int = 0,
) -> List[ProcessedEntry]:
    """Process natural corpus shards without LLM generation.

    Reads each shard, creates a single ProcessedEntry per pair with the given
    instruction, and writes to sharded JSONL output with checkpoint/resume support.

    Args:
        input_paths: List of input JSONL shard paths.
        output_file: Base path for processed JSONL output.
        intermediates_dir: Cache directory for checkpoint files.
        source_lang: Source language code.
        target_lang: Target language code.
        instruction: Fixed instruction string for every entry.
        shard_size: Number of entries per output shard. Default 10000.
        start_shard: Shard index to start writing from. Default 0.

    Returns:
        List of ProcessedEntry dicts.
    """
    stem = output_file.stem
    output_parent = output_file.parent
    checkpoint_parent = intermediates_dir

    output_parent.mkdir(parents=True, exist_ok=True)
    checkpoint_parent.mkdir(parents=True, exist_ok=True)

    processed_hashes: set[str] = set()
    for checkpoint_shard in sorted(checkpoint_parent.glob("*.ckpt")):
        with checkpoint_shard.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_hashes.add(line.strip())

    already_output = len(processed_hashes)
    if already_output > 0:
        print(f"[RESUME] Found {already_output} already-processed pairs")

    all_entries: List[ProcessedEntry] = []
    total_pairs = 0
    skipped = 0

    for shard_path in input_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                prompt = pair.get("prompt", "").strip()
                completion = pair.get("completion", "").strip()
                key = f"{prompt}:{completion}"
                if _md5(key) in processed_hashes:
                    skipped += 1
                    continue
                total_pairs += 1

    print(f"[NATURAL] Processing {total_pairs} pairs (instruction: '{instruction}')")
    if skipped:
        print(f"[RESUME] Skipped {skipped} already-processed pairs")

    def open_shard(idx: int):
        out = output_parent / f"{stem}_{idx:05d}.jsonl"
        ckpt = checkpoint_parent / f"{stem}_{idx:05d}.ckpt"
        return out.open("a", encoding="utf-8"), ckpt.open("a", encoding="utf-8")

    current_shard_idx = start_shard
    pairs_in_shard = 0
    f_out, f_ckpt = open_shard(current_shard_idx)

    pbar = tqdm(total=total_pairs, desc="Natural processing", unit="pair")

    for shard_path in input_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                prompt = pair.get("prompt", "").strip()
                completion = pair.get("completion", "").strip()
                key = f"{prompt}:{completion}"
                pair_hash = _md5(key)

                if pair_hash in processed_hashes:
                    continue

                entry = ProcessedEntry(
                    source=source_lang,
                    target=target_lang,
                    instruction=instruction,
                    input=prompt,
                    output=completion,
                    method="natural",
                )
                all_entries.append(entry)

                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_ckpt.write(pair_hash + "\n")
                pairs_in_shard += 1
                pbar.update(1)

                if pairs_in_shard >= shard_size:
                    f_out.close()
                    f_ckpt.close()
                    current_shard_idx += 1
                    pairs_in_shard = 0
                    f_out, f_ckpt = open_shard(current_shard_idx)

    f_out.close()
    f_ckpt.close()
    pbar.close()

    return all_entries
