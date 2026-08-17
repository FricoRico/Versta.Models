"""Train the glyph-matte U-Net on glyph strips.

Ported (approach, not files) from the MIT-licensed reference implementation by
David Ventura: translator-rs/scripts/ink_model/train.py.

  https://github.com/DavidVentura/translator-rs/blob/master/scripts/ink_model/train.py

Data flow: spawn-context DataLoader workers decode whole batches of labelled
strips from the materialized parquet shards (`MaterializedStrips`
IterableDataset, `collate_fn=identity`); each batch is reused `reuse` steps on
GPU. Labels stay clean; only the RGB is degraded.

CLI entry point is the package `__main__`: `uv run python -m versta.train.glyphmatte`
"""

import random
import shutil
import time

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from ...dataset.glyphmatte.gen_data import Strip
from ...dataset.glyphmatte.parquet import read_shard
from .config import LOSS_WEIGHTS, TrainDefaults
from .model import CompatUNet, GlyphMatteUNet, Outputs, param_count


def _identity(x: object) -> object:
    return x


class MaterializedStrips(IterableDataset):
    """Batches decoded from the materialized parquet shards.

    Workers partition shards by modulo, decode PNG payloads (1-2 ms/strip) and
    bucket strips by width so each yielded batch has a single width. Optionally
    capped for validation; infinite for training (multi-epoch by design).
    """

    def __init__(
        self,
        shards: List[Path],
        batch: int,
        seed: int = 0,
        max_batches: int = 0,
    ):
        super().__init__()
        self.shards = shards
        self.batch = batch
        self.seed = seed
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        mine = [s for i, s in enumerate(self.shards) if i % nw == wid]
        if not mine:
            # Fewer shards than workers: idle workers duplicate the full set
            # instead of spinning without ever filling a batch.
            mine = list(self.shards)
        rng = random.Random(self.seed + wid)
        buckets: Dict[int, List[Strip]] = {}
        yielded = 0
        while self.max_batches <= 0 or yielded < self.max_batches:
            order = mine[:]
            rng.shuffle(order)
            for shard in order:
                for strip in read_shard(shard):
                    w = strip.rgb.shape[1]
                    bucket = buckets.setdefault(w, [])
                    bucket.append(strip)
                    if len(bucket) >= self.batch:
                        sel, buckets[w] = bucket[: self.batch], bucket[self.batch :]
                        yield (
                            torch.from_numpy(
                                np.stack([_strip_to_img(s, w) for s in sel])
                            ),
                            torch.from_numpy(
                                np.stack([_strip_to_label(s) for s in sel])
                            ),
                        )
                        yielded += 1


def _strip_to_img(strip: Strip, width: int) -> np.ndarray:
    rgb = strip.rgb
    if rgb.shape[1] != width:
        raise ValueError("width mismatch")
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)


def _strip_to_label(strip: Strip) -> np.ndarray:
    # [matte, weight, foreground_rgb, background_rgb]
    return np.concatenate(
        [
            strip.matte[None],
            strip.weight[None],
            strip.fcol.transpose(2, 0, 1),
            strip.bcol.transpose(2, 0, 1),
        ]
    ).astype(np.float32)


def losses(pred: Outputs, tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Losses on per-pixel masks/fields; tgt is Bx8xHxW in label order."""
    cov = tgt[:, 0:1]
    out: Dict[str, torch.Tensor] = {}

    sm = torch.sigmoid(pred.matte)
    out["dice"] = 1 - (2 * (sm * cov).sum() + 1.0) / (sm.sum() + cov.sum() + 1.0)
    out["matte_bce"] = F.binary_cross_entropy_with_logits(pred.matte, cov)

    # Weight learned everywhere ink is partially covered (soft coverage mask).
    wmask = cov.clamp(0, 1)
    if float(wmask.sum()) == 0:
        out["weight"] = pred.weight.sum() * 0.0
    else:
        out["weight"] = (wmask * (pred.weight - tgt[:, 1:2]).abs()).sum() / wmask.sum()

    # Colours: foreground signed by the ink, background by the paper. The
    # labels degrade in lockstep with the RGB through `photometric_aux`, so
    # sigmoid(pred) stays in range for a plain L1.
    f_t = tgt[:, 2:5]
    b_t = tgt[:, 5:8]
    fg = torch.sigmoid(pred.foreground)
    bg = torch.sigmoid(pred.background)
    f_mask = torch.nan_to_num(cov).clamp(0, 1)
    b_mask = 1.0 - f_mask
    out["color"] = (f_mask * (fg - f_t).abs()).sum() / (f_mask.sum() * 3).clamp(
        min=1
    ) + (b_mask * (bg - b_t).abs()).sum() / (b_mask.sum() * 3).clamp(min=1)
    return out


def run_val(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for img, tgt in loader:
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            pred = model(img)
            # Per-strip matte IoU at 0.5 (same convention as eval.py).
            pm = torch.sigmoid(pred.matte[:, 0]) > 0.5
            gt = tgt[:, 0] > 0.5
            denom = (pm | gt).sum(dim=(1, 2)).float() + 1e-6
            iou = ((pm & gt).sum(dim=(1, 2)) / denom).mean()
            wmask = (tgt[:, 0:1] > 0.5).float()
            weight_mae = (wmask * (pred.weight - tgt[:, 1:2]).abs()).sum() / (
                wmask.sum() + 1e-6
            )
            cov3 = wmask.repeat(1, 3, 1, 1)
            bg3 = 1.0 - cov3
            fg_mae = (
                cov3 * (torch.sigmoid(pred.foreground) - tgt[:, 2:5]).abs()
            ).sum() / (cov3.sum() + 1e-6)
            bg_mae = (
                bg3 * (torch.sigmoid(pred.background) - tgt[:, 5:8]).abs()
            ).sum() / (bg3.sum() + 1e-6)
            sums["iou"] = sums.get("iou", 0.0) + float(iou)
            sums["weight_mae"] = sums.get("weight_mae", 0.0) + float(weight_mae)
            sums["fg_mae"] = sums.get("fg_mae", 0.0) + float(fg_mae)
            sums["bg_mae"] = sums.get("bg_mae", 0.0) + float(bg_mae)
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def save_ckpt(model: nn.Module, step: int, path: Path) -> None:
    raw = model
    while hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": raw.state_dict(),
            "config": {
                "base": getattr(raw, "_base", 16),
                "levels": raw.levels,
                "weight_head": raw.weight_head_kind,
            },
        },
        path,
    )


def load_for_resume(ckpt: Path, model: nn.Module) -> int:
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = blob["state_dict"]
    if not any(k.startswith("weight_head") for k in sd):
        sd = CompatUNet.remap_state_dict(sd)
    missing, unexpected = nn.Module.load_state_dict(model, sd, strict=False)
    if missing or unexpected:
        print(f"resume remap: missing={missing} unexpected={unexpected}", flush=True)
    return int(blob.get("step", 0))


def train_model(cfg: TrainDefaults, output_dir: Path, resume: Optional[Path]) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(
        "cuda"
        if (cfg.device == "auto" and torch.cuda.is_available())
        else (cfg.device if cfg.device != "auto" else "cpu")
    )
    base, levels = (int(x) for x in cfg.size.split("x"))

    model = GlyphMatteUNet(base=base, levels=levels, weight_head=cfg.weight_head)
    model._base = base
    start_step = 0
    if resume:
        start_step = load_for_resume(resume, model)
        print(f"resumed {resume} @ step {start_step}", flush=True)
    model = model.to(device)
    n_params = param_count(model)
    print(
        f"device={device} params={n_params:,} int8~={n_params / 1e6:.2f}MB", flush=True
    )

    have_cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cfg.compile and device.type == "cuda" and have_cc:
        model = torch.compile(model)
    elif cfg.compile and not have_cc:
        print("torch.compile enabled but no C compiler; running eager", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(cfg.steps, 100), pct_start=0.05
    )

    ckpt_dir = output_dir / "intermediates" / "ckpt"
    dataset_dir = output_dir / "intermediates" / "dataset"

    def make_loader() -> DataLoader:
        shards = sorted(dataset_dir.glob("train-*.parquet"))
        if not shards:
            raise RuntimeError(
                f"no train shards in {dataset_dir}; the dataset stage must run first"
            )
        ds = MaterializedStrips(shards, cfg.batch, cfg.seed)
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=cfg.workers,
            collate_fn=_identity,
            pin_memory=True,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    val_shards = sorted(dataset_dir.glob("validation-*.parquet"))
    val_ds = MaterializedStrips(
        val_shards, cfg.batch, cfg.seed + 999, max(cfg.val // cfg.batch, 1)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=2,
        collate_fn=_identity,
        multiprocessing_context="spawn",
    )

    train_loader = None

    model.train()
    t0 = time.time()
    seen = 0
    step = start_step
    running: Dict[str, float] = {}

    while step < cfg.steps:
        if train_loader is None:
            train_loader = make_loader()
            it = iter(train_loader)

        img_t, tgt_t = next(it)

        # Same-width pad wrapper: everything reaches the GPU at wrapper width
        # so torch.compile sees a constant shape.
        w_now = img_t.shape[3]
        if w_now < cfg.wrapper_width:
            img_t = F.pad(img_t, (0, cfg.wrapper_width - w_now))
            tgt_t = F.pad(tgt_t, (0, cfg.wrapper_width - w_now))

        img_t = img_t.to(device, non_blocking=True)
        tgt_t = tgt_t.to(device, non_blocking=True)

        for _ in range(cfg.reuse):
            pred = model(img_t)
            ls = losses(pred, tgt_t)
            loss = sum(LOSS_WEIGHTS[k] * v for k, v in ls.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in ls.items():
                running[k] = running.get(k, 0.0) + float(v) / 25
            step += 1
            seen += img_t.shape[0]
            if step % 25 == 0:
                el = time.time() - t0
                msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(running.items()))
                print(f"step {step} {msg} ({seen / el:.0f}/s)", flush=True)
                running = {}
            if step % 500 == 0:
                if step % 4000 == 0 and step > 0:
                    save_ckpt(model, step, ckpt_dir / f"glyphmatte-step{step}.pt")
                save_ckpt(model, step, ckpt_dir / "glyphmatte-latest.pt")
                val = run_val(model, val_loader, device)
                print(
                    f"step {step} VAL {val} lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if step >= cfg.steps:
                break

    save_ckpt(model, step, ckpt_dir / f"glyphmatte-step{step}.pt")
    save_ckpt(model, step, ckpt_dir / "glyphmatte-latest.pt")
    print("done", flush=True)
