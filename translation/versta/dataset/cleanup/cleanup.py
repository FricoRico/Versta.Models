import hashlib
import json
from pathlib import Path


def _md5(text: str) -> int:
    return int.from_bytes(
        hashlib.md5(text.encode("utf-8")).digest(), "big"
    )


def _scan(intermediates_dir: Path) -> tuple[set[int], set[int]]:
    pair_hashes: set[int] = set()
    input_hashes: set[int] = set()

    patterns = [
        "coordinated/*.jsonl",
        "synthetic/*.filtered.jsonl",
        "natural/**/*.filtered.jsonl",
    ]

    for pattern in patterns:
        for f in sorted(intermediates_dir.glob(pattern)):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
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
                    pair_hashes.add(_md5(f"{prompt}:{completion}"))
                    pair_hashes.add(_md5(f"{completion}:{prompt}"))
                    input_hashes.add(_md5(prompt))
                    input_hashes.add(_md5(completion))

    return pair_hashes, input_hashes


def _find_checkpoint_dirs(intermediates_dir: Path) -> list[Path]:
    dirs: set[Path] = set()
    for ckpt in intermediates_dir.rglob("*.ckpt"):
        dirs.add(ckpt.parent)
    return sorted(dirs)


def _clean_checkpoints(
    checkpoint_dirs: list[Path],
    valid_hashes: set[int],
    intermediates_dir: Path,
    dry_run: bool = False,
) -> dict:
    stats = {"files": 0, "removed": 0, "kept": 0}

    for ckpt_dir in checkpoint_dirs:
        for ckpt_file in sorted(ckpt_dir.glob("*.ckpt")):
            stats["files"] += 1
            lines = [
                l.strip()
                for l in ckpt_file.read_text().splitlines()
                if l.strip()
            ]
            kept = []
            removed = 0
            for line in lines:
                h = int(line, 16)
                if h in valid_hashes:
                    kept.append(line)
                else:
                    removed += 1
            stats["removed"] += removed
            stats["kept"] += len(kept)

            if removed > 0:
                rel = ckpt_file.relative_to(intermediates_dir.parent.parent)
                print(f"  CLEAN {rel}: {removed} stale hashes ({len(kept)} kept)")
            if removed > 0 and not dry_run:
                ckpt_file.write_text(
                    "\n".join(kept) + "\n" if kept else ""
                )

    return stats


def _clean_output_shards(
    output_dir: Path,
    pair_hashes: set[int],
    input_hashes: set[int],
    dry_run: bool = False,
) -> dict:
    stats = {
        "files": 0,
        "natural_removed": 0,
        "synthetic_removed": 0,
        "kept": 0,
    }
    seen_hashes: set[int] = set()

    for shard_file in sorted(output_dir.glob("dataset_*.jsonl")):
        stats["files"] += 1
        lines = shard_file.read_text().splitlines()
        kept: list[str] = []
        natural_removed = 0
        synthetic_removed = 0

        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue

            inp = entry.get("input", "").strip()
            out = entry.get("output", "").strip()
            method = entry.get("method", "")

            if not inp or not out:
                kept.append(line)
                continue

            if method == "natural":
                dedup_key = _md5(f"{inp}:{out}:{entry.get('instruction', '')}")
                if dedup_key in seen_hashes:
                    natural_removed += 1
                    continue
                h = _md5(f"{inp}:{out}")
                if h in pair_hashes:
                    seen_hashes.add(dedup_key)
                    kept.append(line)
                else:
                    natural_removed += 1
            elif method == "synthetic":
                dedup_key = _md5(f"{inp}:{out}:{entry.get('instruction', '')}")
                if dedup_key in seen_hashes:
                    synthetic_removed += 1
                    continue
                h = _md5(inp)
                if h in input_hashes:
                    seen_hashes.add(dedup_key)
                    kept.append(line)
                else:
                    synthetic_removed += 1
            else:
                kept.append(line)

        stats["natural_removed"] += natural_removed
        stats["synthetic_removed"] += synthetic_removed
        stats["kept"] += len(kept)

        total_removed = natural_removed + synthetic_removed
        if total_removed > 0:
            print(
                f"  SHARD {shard_file.name}: "
                f"{natural_removed} natural, {synthetic_removed} synthetic removed"
                f" ({len(kept)} kept)"
            )
            if not dry_run:
                shard_file.write_text("\n".join(kept) + "\n")

    return stats


def cleanup_lang_pair(
    pair_dir: Path,
    dry_run: bool = False,
) -> dict:
    print(f"\n=== {pair_dir.name} ===")

    intermediates_dir = pair_dir / "intermediates"
    if not intermediates_dir.exists():
        print("  No intermediates directory, skipping")
        return {}

    print("  Building hash indexes from intermediates...")
    pair_hashes, input_hashes = _scan(intermediates_dir)
    print(
        f"  {len(pair_hashes)} pair hashes, "
        f"{len(input_hashes)} input hashes"
    )

    checkpoint_dirs = _find_checkpoint_dirs(intermediates_dir)
    if checkpoint_dirs:
        print(
            f"  Cleaning {len(checkpoint_dirs)} checkpoint directories..."
        )
        ckpt_stats = _clean_checkpoints(
            checkpoint_dirs, pair_hashes, intermediates_dir, dry_run
        )
    else:
        ckpt_stats = {"files": 0, "removed": 0, "kept": 0}
        print("  No checkpoint files found")

    out_stats: dict = {"files": 0, "natural_removed": 0, "synthetic_removed": 0, "kept": 0}
    if list(pair_dir.glob("dataset_*.jsonl")):
        print("  Cleaning output shards...")
        out_stats = _clean_output_shards(
            pair_dir, pair_hashes, input_hashes, dry_run
        )
    else:
        print("  No output shards found")

    summary = {
        "pair": pair_dir.name,
        "checkpoints": ckpt_stats,
        "output": out_stats,
    }

    total_removed = (
        ckpt_stats["removed"]
        + out_stats["natural_removed"]
        + out_stats["synthetic_removed"]
    )
    kept_total = ckpt_stats["kept"] + out_stats["kept"]
    print(
        f"  Summary: {total_removed} total removed, "
        f"{kept_total} kept"
    )

    return summary


def cleanup_all(
    base_output: Path,
    dry_run: bool = False,
) -> list[dict]:
    results = []
    for pair_dir in sorted(base_output.iterdir()):
        if pair_dir.is_dir():
            result = cleanup_lang_pair(pair_dir, dry_run)
            if result:
                results.append(result)
    return results
