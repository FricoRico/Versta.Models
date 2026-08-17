"""Parquet shard writer/reader for the glyphmatte strip dataset.

Storage: `rgb`, `foreground`, `background` as RGB PNG bytes; `matte`, `weight`
as single-channel PNG bytes (uint8 = v*255). HF auto-detects `train-*`/
`validation-*` parquet files at the repo root, so
`datasets.load_dataset(...)` works without a loader script; our own reader
uses pyarrow directly to keep the training deps light.
"""

import io

from pathlib import Path
from typing import Iterator, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .config import PARQUET_COLUMNS
from .gen_data import Strip


def _png_u8(arr: np.ndarray) -> bytes:
    """H,W float 0..1 -> L PNG bytes; H,W,3 float 0..1 -> RGB PNG bytes."""
    if arr.ndim == 2:
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    else:
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _from_png_u8(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        return np.ascontiguousarray(arr)
    return arr


def write_shard(strips: List[Strip], path: Path) -> None:
    """Writes one parquet shard. Keeps natural width per strip — PNG payloads
    carry the geometry; parquet rows are width-heterogeneous by design.

    Args:
        strips (List[Strip]): The batch of strips for this shard.
        path (Path): Target .parquet path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            col: [_png_u8(getattr_strip(s, col)) for s in strips]
            for col in PARQUET_COLUMNS
        }
    )
    pq.write_table(table, path, compression="zstd")


def getattr_strip(strip: Strip, name: str) -> np.ndarray:
    m = {
        "rgb": strip.rgb,
        "matte": strip.matte,
        "weight": strip.weight,
        "foreground": strip.fcol,
        "background": strip.bcol,
    }
    return m[name]


def read_strip_row(row: dict) -> Strip:
    """One parquet row -> Strip (arrays back in float32 0..1)."""
    rgb = _from_png_u8(row["rgb"])
    matte = _from_png_u8(row["matte"])
    weight = _from_png_u8(row["weight"])
    fcol = _from_png_u8(row["foreground"])
    bcol = _from_png_u8(row["background"])
    return Strip(rgb=rgb, matte=matte, weight=weight, fcol=fcol, bcol=bcol)


def read_shard(path: Path, batch: int = 256) -> Iterator[Strip]:
    """Streams strips from one shard in row-group-friendly batches."""
    f = pq.ParquetFile(path)
    for rg in f.iter_batches(batch_size=batch, columns=list(PARQUET_COLUMNS)):
        d = rg.to_pydict()
        for i in range(rg.num_rows):
            yield read_strip_row({c: d[c][i] for c in PARQUET_COLUMNS})


def read_shards(paths: List[Path], batch: int = 256) -> Iterator[Strip]:
    for path in paths:
        yield from read_shard(path, batch)
