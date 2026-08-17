"""Materializes the synthetic glyphmatte strip dataset to parquet shards.

Writes sharded parquet (HF-convention names `train-00000-of-0000N.parquet` /
`validation-00000-of-00001.parquet`, so the official `datasets` package loads
the repo without a loader script) plus a shared `metadata.jsonl` with one
traceability row per strip (text, font, native height, weight target, fg/bg
colors, …).
"""

import json
import random

from pathlib import Path
from typing import List

from tqdm import tqdm

from . import assets as assets_mod
from .config import DATASET, STRIP
from .gen_data import Strip, sample_strip
from .parquet import write_shard


def metadata_row(strip: Strip, index: int, shard: str, row: int) -> dict:
    """One metadata.jsonl row per strip; colors from generator params (base
    fg triple; gradient ink also records `foreground_rgb_end`)."""
    p = strip.params
    return {
        "index": index,
        "shard": shard,
        "row": row,
        "text": p["text"],
        "script": p["script"],
        "font": p["font"],
        "native_h": p["native_h"],
        "stroke_w": p["stroke_w"],
        "stroke_ratio": p["stroke_ratio"],
        "weight_target": p["weight_target"],
        "foreground_rgb": p.get("foreground_rgb"),
        "foreground_rgb_end": p.get("foreground_rgb_end"),
        "background_rgb": p.get("background_rgb"),
        "width": int(strip.rgb.shape[1]),
        "height": STRIP.height,
    }


def materialize_split(
    out: Path,
    split: str,
    n: int,
    shard_size: int,
    rng: random.Random,
    index_offset: int = 0,
) -> List[str]:
    """Writes `n` strips as `<split>-XXXXX-of-YYYYY.parquet` shards.

    Returns the metadata JSON lines for this split.

    Args:
        out (Path): Target directory.
        split (str): "train" or "validation" (HF name convention).
        n (int): Strip count.
        shard_size (int): Strips per shard.
        rng (random.Random): Shared RNG so shard content only depends on seed.
        index_offset (int): Global index of this split's first strip.

    Returns:
        List[str]: One JSON line per strip.
    """
    shards = max(1, (n + shard_size - 1) // shard_size)
    rows: List[str] = []
    strips: List[Strip] = []
    for i in tqdm(range(n), desc=split):
        strips.append(
            sample_strip(rng, width=rng.choice(STRIP.widths), keep_degrade=True)
        )
        if len(strips) == shard_size or i == n - 1:
            shard_idx = i // shard_size
            shard = f"{split}-{shard_idx:05d}-of-{shards:05d}.parquet"
            write_shard(strips, out / shard)
            base = index_offset + shard_idx * shard_size
            for local_row, strip in enumerate(strips):
                rows.append(
                    json.dumps(metadata_row(strip, base + local_row, shard, local_row))
                )
            strips = []
    return rows


def materialize_dataset(
    out: Path,
    n: int = DATASET.n,
    val_n: int = DATASET.val_n,
    shard_size: int = DATASET.shard_size,
    seed: int = DATASET.seed,
) -> None:
    """Full materialization: train shards + validation shard + metadata.jsonl.

    Syncs the pinned fonts/word lists on first use (idempotent: existing
    files are skipped).
    """
    if missing := assets_mod.missing_assets():
        print(f"syncing {len(missing)} pinned assets...")
        assets_mod.sync_assets()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = materialize_split(out, "train", n, shard_size, rng, 0)
    rows += materialize_split(out, "validation", val_n, max(val_n, shard_size), rng, n)
    (out / "metadata.jsonl").write_text("\n".join(rows) + "\n")
    print(f"wrote {out} ({n} train + {val_n} validation strips)")
