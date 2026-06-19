import hashlib
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from opustools import OpusRead

from .types import ExtractionResult


def create_reversed_shards(
    input_shards: list[Path], intermediates_dir: Path
) -> list[Path]:
    """Swap prompt/completion in each shard, write reversed shards to disk.

    Args:
        input_shards (list[Path]): List of input JSONL shard paths.
        intermediates_dir (Path): Directory to store reversed shards.

    Returns:
        list[Path]: List of paths to the reversed shard files.
    """
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


def download_opus_dataset(
    source: str,
    target: str,
    download_dir: Path,
    intermediates_dir: Path,
    corpus: str,
    pairs: int | None = None,
    release: str | None = None,
    preprocess: str = "raw",
) -> ExtractionResult:
    """Download parallel sentence pairs from OPUS (opus.nlpl.eu) for a given language pair.

    Uses OpusRead with moses write mode to extract clean sentence pairs
    into a JSONL file — one entry per line with 'prompt' and 'completion' keys (TRL format).

    Args:
        source: Source language code (e.g. 'en').
        target: Target language code (e.g. 'es').
        download_dir: Directory to store downloaded corpus text files.
        intermediates_dir: Directory to store intermediate JSONL files.
        corpus: OPUS corpus name (e.g. 'OpenSubtitles', 'CCMatrix', 'Europarl').
        pairs: Maximum number of sentence pairs to extract. None for all.
        release: Version of corpus to download.
        preprocess: OPUS preprocessing type (e.g. 'raw' or 'moses'). Default 'raw'.

    Returns:
        Dict with keys: 'source', 'target', 'corpus', 'num_pairs', 'output_file'.

    Raises:
        ModuleNotFoundError: If opustools is not installed.
        RuntimeError: If download or extraction fails, or no pairs are found.
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    src_out = download_dir / f"{corpus}_{source}.txt"
    tgt_out = download_dir / f"{corpus}_{target}.txt"

    args = {
        "directory": corpus,
        "source": source,
        "target": target,
        "preprocess": preprocess,
        "write_mode": "moses",
        "write": [str(src_out), str(tgt_out)],
        "suppress_prompts": True,
        "leave_non_alignments_out": True,
    }

    if release is not None:
        args["release"] = release

    if pairs is not None and preprocess == "raw":
        args["maximum"] = pairs

    args["download_dir"] = str(download_dir)

    if src_out.exists() and tgt_out.exists():
        print(f"Using cached extraction for {corpus}")
    else:
        opus_reader = OpusRead(**args)
        opus_reader.printPairs()

    if not src_out.exists() or not tgt_out.exists():
        if src_out.exists():
            src_out.unlink()
        if tgt_out.exists():
            tgt_out.unlink()
        raise RuntimeError(
            f"Failed to create output files for corpus '{corpus}', "
            f"language pair '{source}-{target}'. "
            "Check the corpus name and language codes are correct."
        )

    output_file = intermediates_dir / f"{corpus}_{source}-{target}.jsonl"
    num_pairs = 0

    t = tqdm(total=pairs, unit="pair", desc=f"Converting {corpus}")

    with (
        src_out.open("r", encoding="utf-8") as src_f,
        tgt_out.open("r", encoding="utf-8") as tgt_f,
        output_file.open("w", encoding="utf-8") as out_f,
    ):
        for src_line, tgt_line in zip(src_f, tgt_f):
            src = src_line.strip()
            tgt = tgt_line.strip()
            if not src or not tgt:
                continue
            if pairs is not None and num_pairs >= pairs:
                break

            out_f.write(
                json.dumps(
                    {"prompt": src, "completion": tgt},
                    ensure_ascii=False,
                )
                + "\n"
            )
            num_pairs += 1
            t.update(1)

    t.close()

    if num_pairs == 0:
        raise RuntimeError(
            f"No sentence pairs found for corpus '{corpus}', "
            f"language pair '{source}-{target}'. "
            "Check the corpus name and language codes are correct."
        )

    return ExtractionResult(
        source=source,
        target=target,
        corpus=corpus,
        num_pairs=num_pairs,
        output_file=str(output_file),
    )


def smart_sample(
    jsonl_path: str,
    output_path: Path,
    pairs: int | None = None,
    min_chars: int = 5,
    max_chars: int = 500,
    seed: int = 42,
) -> dict:
    """Apply quality filters and deterministic sampling to a JSONL dataset.

    Filters applied in order:
        1. Remove lines with empty src/tgt
        2. Filter by character length (5-500 chars default)
        3. Deduplicate by MD5 hash of src+tgt
        4. Deterministic random sample if target_size is specified

    Args:
        jsonl_path: Path to the input JSONL file.
        output_path: Where to write the filtered and sampled JSONL.
        pairs: Maximum number of pairs to keep after filtering.
        min_chars: Minimum character count for source and target text.
        max_chars: Maximum character count for source and target text.
        seed: Random seed for deterministic sampling.

    Returns:
        Dict with keys: 'raw_count', 'filtered_count', 'kept_count'.
    """
    rng = random.Random(seed)
    seen_hashes = set()
    valid_pairs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = pair.get("prompt", "").strip()
            completion = pair.get("completion", "").strip()

            if not prompt or not completion:
                continue

            if not (min_chars <= len(prompt) <= max_chars):
                continue
            if not (min_chars <= len(completion) <= max_chars):
                continue

            pair_hash = hashlib.md5(
                f"{prompt}:{completion}".encode("utf-8")
            ).hexdigest()
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)

            valid_pairs.append(pair)

    final_pairs = valid_pairs
    if pairs is not None and len(valid_pairs) > pairs:
        final_pairs = rng.sample(valid_pairs, pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for pair in final_pairs:
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")

    result = {
        "raw_count": len(valid_pairs),
        "filtered_count": len(valid_pairs) - len(final_pairs),
        "kept_count": len(final_pairs),
    }

    return result


def merge_and_dedup(
    filtered_paths: list[str],
    filtered_file_path: Path,
    shard_size: int = 10000,
) -> dict:
    """Merge multiple filtered JSONL files and remove duplicate pairs across corpora.

    Takes all pairs from the provided JSONL files, deduplicates by MD5 hash of src+tgt,
    and writes the merged result into sharded files: output_00000.jsonl, output_00001.jsonl, etc.

    Args:
        filtered_paths (list[str]): List of paths to filtered JSONL files.
        filtered_file_path (Path): Base path for the merged and deduplicated JSONL shards.
        shard_size (int): Number of pairs per shard. Default 10000.

    Returns:
        dict: Dict with keys: 'total' (total pairs across all files), 'kept' (pairs after dedup),
            'duplicates_removed', 'shard_count' (number of shards created).
    """
    hashes = set()
    current_shard_pairs = []
    shard_count = 0
    total_count = 0

    for path in filtered_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_count += 1
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt = pair.get("prompt", "").strip()
                completion = pair.get("completion", "").strip()

                if not prompt or not completion:
                    continue

                pair_hash = hashlib.md5(
                    f"{prompt}:{completion}".encode("utf-8")
                ).hexdigest()
                if pair_hash in hashes:
                    continue
                hashes.add(pair_hash)

                current_shard_pairs.append(pair)

                if len(current_shard_pairs) >= shard_size:
                    shard_name = f"{filtered_file_path.stem}_{shard_count:05d}.jsonl"
                    shard_path = filtered_file_path.parent / shard_name
                    with shard_path.open("w", encoding="utf-8") as f_out:
                        for pair in current_shard_pairs:
                            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    current_shard_pairs.clear()
                    shard_count += 1

    if current_shard_pairs:
        shard_name = f"{filtered_file_path.stem}_{shard_count:05d}.jsonl"
        shard_path = filtered_file_path.parent / shard_name
        with shard_path.open("w", encoding="utf-8") as f_out:
            for pair in current_shard_pairs:
                f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
        shard_count += 1

    result = {
        "total": total_count,
        "kept": len(hashes),
        "duplicates_removed": total_count - len(hashes),
        "shard_count": shard_count,
    }

    return result
