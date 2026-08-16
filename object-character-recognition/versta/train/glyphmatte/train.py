"""Train the glyph-matte U-Net on synthetic strips.

Ported (approach, not files) from the MIT-licensed reference implementation by
David Ventura: translator-rs/scripts/ink_model/train.py.

  https://github.com/DavidVentura/translator-rs/blob/master/scripts/ink_model/train.py

Data flow: spawn-context DataLoader workers generate whole batches of labelled
strips (`SyntheticStrips` IterableDataset, `collate_fn=identity`) so the burst
batch B is generated ONCE per worker cycle and reused `reuse` steps on GPU.
Labels stay clean; only the RGB is degraded.

CLI entry point is the package `__main__`: `uv run python -m versta.train.glyphmatte`
"""

import argparse
import random
import time

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .gen_data import WIDTHS, Strip, sample_strip
from .model import CompatUNet, GlyphMatteUNet, Outputs, param_count

CKPT_DIR = Path("cache/checkpoints")


def _identity(x: object) -> object:
    return x


class SyntheticStrips(IterableDataset):
    """Infinite stream of synthetic strip batches; each worker yield IS a
    whole batch (`collate_fn=identity`). Val mode caps at val_steps batches.

    Modes: "train" (gradual degrade prob ramp) / "val" (fixed degrade prob 1).
    """

    def __init__(
        self,
        batch: int,
        mode: str,
        val_steps: int,
        degrade_prob: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.batch = batch
        self.mode = mode
        self.val_steps = val_steps
        self.degrade_prob = degrade_prob
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        n_yield = self.val_steps if self.mode == "val" else None
        i = 0
        while n_yield is None or i < n_yield:
            width = rng.choice(WIDTHS)
            imgs, labels = [], []
            for _ in range(self.batch):
                deg = rng.random() < self.degrade_prob
                strip = sample_strip(rng, width=width, keep_degrade=deg)
                imgs.append(_strip_to_img(strip, width))
                labels.append(_strip_to_label(strip))
            yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labels))
            i += 1


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


LOSS_WEIGHTS: Dict[str, float] = {
    "dice": 1.0,
    "matte_bce": 1.0,
    "weight": 3.0,
    "color": 1.0,
}


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


def train_model(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    base, levels = (int(x) for x in args.size.split("x"))

    model = GlyphMatteUNet(base=base, levels=levels, weight_head=args.weight_head)
    model._base = base
    start_step = 0
    if args.resume:
        start_step = load_for_resume(args.resume, model)
        print(f"resumed {args.resume} @ step {start_step}", flush=True)
    model = model.to(device)
    n_params = param_count(model)
    print(
        f"device={device} params={n_params:,} int8~={n_params / 1e6:.2f}MB", flush=True
    )

    import shutil

    have_cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if args.compile and device.type == "cuda" and have_cc:
        model = torch.compile(model)
    elif args.compile and not have_cc:
        print("torch.compile requested but no C compiler; running eager", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.steps, 100), pct_start=0.05
    )

    def degrade_prob(step: int) -> float:
        return min(1.0, step / max(args.degrade_ramp, 1))

    def make_loader(seed_off: int, deg: float) -> DataLoader:
        ds = SyntheticStrips(args.batch, "train", 0, deg, args.seed + seed_off)
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=args.workers,
            collate_fn=_identity,
            pin_memory=True,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    # Val set: fixed seed, always fully degraded; 2 workers, no persistence.
    val_ds = SyntheticStrips(
        args.batch, "val", max(args.val // args.batch, 1), 1.0, args.seed + 999
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

    while step < args.steps:
        if step % 500 == 0 or train_loader is None:
            deg = degrade_prob(step)
            train_loader = make_loader(step // 500, deg)
            it = iter(train_loader)

        img_t, tgt_t = next(it)

        # Same-width pad wrapper: everything reaches the GPU at wrapper width
        # so torch.compile sees a constant shape.
        w_now = img_t.shape[3]
        if w_now < args.wrapper_width:
            img_t = F.pad(img_t, (0, args.wrapper_width - w_now))
            tgt_t = F.pad(tgt_t, (0, args.wrapper_width - w_now))

        img_t = img_t.to(device, non_blocking=True)
        tgt_t = tgt_t.to(device, non_blocking=True)

        for _ in range(args.reuse):
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
                    save_ckpt(model, step, CKPT_DIR / f"glyphmatte-step{step}.pt")
                save_ckpt(model, step, CKPT_DIR / "glyphmatte-latest.pt")
                val = run_val(model, val_loader, device)
                print(
                    f"step {step} VAL {val} lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if step >= args.steps:
                break

    save_ckpt(model, step, CKPT_DIR / f"glyphmatte-step{step}.pt")
    save_ckpt(model, step, CKPT_DIR / "glyphmatte-latest.pt")
    print("done", flush=True)
