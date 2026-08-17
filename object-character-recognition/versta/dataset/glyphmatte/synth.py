"""Shared camera/screen degradation and legibility gating for ink synthesis.

Ported (approach, not files) from the MIT-licensed reference implementation by
David Ventura: translator-rs/scripts/synth_core.py. The degrade path runs on
the CPU per-strip; only the composited image is degraded — labels stay clean.

  https://github.com/DavidVentura/translator-rs/blob/master/scripts/synth_core.py
"""

import io
import random

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


@lru_cache(maxsize=64)
def coord_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read-only (yy, xx) float32 pixel grids, cached per size."""
    yy, xx = np.mgrid[0:h, 0:w]
    yy = np.ascontiguousarray(yy, dtype=np.float32)
    xx = np.ascontiguousarray(xx, dtype=np.float32)
    yy.flags.writeable = False
    xx.flags.writeable = False
    return yy, xx


def random_color(rng: random.Random) -> np.ndarray:
    """A random RGB in 0..1."""
    return np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)


def degrade(
    img: np.ndarray,
    rng: random.Random,
    native_h: int,
    log: Optional[Dict[str, object]] = None,
    photometric_aux: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Camera/screen degradations applied to the composited image only.

    Blur scales with the native text height: a 1.8 px gaussian erases 12 px text
    outright but is realistic camera softness on 40 px text. Labels stay clean.

    `photometric_aux`: HxWx3 label images (the ink/background colour fields,
    when the colour head is trained) updated in place with the colour-changing
    ops only — shade, hard shadow, contrast squeeze. Illumination genuinely
    changes the colour the labels should carry, while blur/JPEG/noise are
    observation noise the model must see through. These ops are per-pixel
    affine, so applying them to the fields independently keeps the compositing
    identity `img ≈ cov·F + (1−cov)·B` exact up to observation noise.
    """
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if rng.random() < 0.5:
        sigma = rng.uniform(0.3, min(0.35 + native_h * 0.032, 2.0))
        pil = pil.filter(ImageFilter.GaussianBlur(sigma))
        if log is not None:
            log["blur"] = round(sigma, 2)
    if native_h >= 24 and rng.random() < 0.25:
        # Crude motion blur: directional box kernel. Skipped on tiny text.
        k = rng.choice([3, 5])
        if log is not None:
            log["motion"] = k
        kernel = [0.0] * (k * k)
        if rng.random() < 0.5:
            for i in range(k):
                kernel[(k // 2) * k + i] = 1.0 / k
        else:
            for i in range(k):
                kernel[i * k + k // 2] = 1.0 / k
        pil = pil.filter(ImageFilter.Kernel((k, k), kernel, scale=1.0))
    if rng.random() < 0.35:
        scale = rng.uniform(0.55, 0.85)
        small = pil.resize(
            (max(8, int(pil.width * scale)), max(8, int(pil.height * scale))),
            Image.BILINEAR,
        )
        pil = small.resize((pil.width, pil.height), Image.BILINEAR)
        if log is not None:
            log["downsample"] = round(scale, 2)
    if rng.random() < 0.7:
        q = rng.randint(45, 95)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        if log is not None:
            log["jpeg"] = q
    out = np.asarray(pil, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        yy, xx = coord_grid(h, w)
        angle = rng.uniform(0, 2 * np.pi)
        t = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        shade = np.clip(rng.uniform(0.55, 1.0) + t * rng.uniform(0.0, 0.45), 0.4, 1.2)
        out = out * shade[..., None]
        if photometric_aux is not None:
            for i, aux in enumerate(photometric_aux):
                photometric_aux[i] = aux * shade[..., None]
        if log is not None:
            log["shade"] = 1
    if rng.random() < 0.25:
        # Hard-edged cast shadow: a strong illumination edge looks like a
        # stroke to the model unless it has trained on shadows that aren't ink.
        h, w = out.shape[:2]
        yy, xx = coord_grid(h, w)
        angle = rng.uniform(0, 2 * np.pi)
        proj = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        edge = rng.uniform(float(proj.min()), float(proj.max()))
        shadow = np.where(proj < edge, rng.uniform(0.4, 0.8), 1.0).astype(np.float32)
        out = out * shadow[..., None]
        if photometric_aux is not None:
            for i, aux in enumerate(photometric_aux):
                photometric_aux[i] = aux * shadow[..., None]
        if log is not None:
            log["hardshadow"] = 1
    if rng.random() < 0.6:
        nsig = rng.uniform(0.005, 0.04)
        out = out + rng_noise(rng, out.shape) * nsig
        if log is not None:
            log["noise"] = round(nsig, 3)
    if rng.random() < 0.3:
        lo, hi = rng.uniform(0.0, 0.08), rng.uniform(0.85, 1.0)
        out = out * (hi - lo) + lo
        if photometric_aux is not None:
            for i, aux in enumerate(photometric_aux):
                photometric_aux[i] = aux * (hi - lo) + lo
        if log is not None:
            log["squeeze"] = round(hi - lo, 2)
    if photometric_aux is not None:
        # Shade's 1.2x ceiling can push a bright field past 1; labels must stay
        # in the sigmoid's range.
        for i, aux in enumerate(photometric_aux):
            photometric_aux[i] = np.clip(aux, 0, 1).astype(np.float32, copy=False)
    return np.clip(out, 0, 1)


def rng_noise(rng: random.Random, shape: Tuple[int, ...]) -> np.ndarray:
    """Deterministic gaussian noise from a random.Random (worker-seedable)."""
    return (
        np.random.default_rng(rng.getrandbits(32))
        .normal(0, 1, shape)
        .astype(np.float32)
    )


def legible(img: np.ndarray, cov: np.ndarray, native_h: int) -> bool:
    """Reject pairs whose degraded ink no longer contrasts with the background.

    Compares median ink color against median background color within the text's
    own rows (global background medians lie on gradient strips).

    Args:
        img (np.ndarray): The degraded strip, HxWx3 in 0..1.
        cov (np.ndarray): Clean glyph coverage, HxW in 0..1.
        native_h (int): Native text height in pixels.

    Returns:
        bool: True if the strip stays legible.
    """
    ink_mask = cov > 0.6
    if ink_mask.sum() < 30:
        return False
    text_rows = ink_mask.any(axis=1)
    bg_mask = (cov < 0.05) & text_rows[:, None]
    if bg_mask.sum() >= 30:
        bg_px = img[bg_mask]
    else:
        # Dense display text: sample the lowest-coverage pixels in the text
        # band (the gaps/counters) as background instead.
        band_cov = np.where(text_rows[:, None], cov, 2.0).ravel()
        k = min(400, max(30, int(text_rows.sum()) * cov.shape[1] // 5))
        k = min(k, band_cov.size - 1)
        if k < 30:
            return False
        idx = np.argpartition(band_cov, k)[:k]
        if band_cov[idx].max() >= 0.6:
            return False
        bg_px = img.reshape(-1, img.shape[-1])[idx]
    ink_px = img[ink_mask]
    if len(ink_px) > 400:
        ink_px = ink_px[:: len(ink_px) // 400]
    if len(bg_px) > 400:
        bg_px = bg_px[:: len(bg_px) // 400]
    d = np.abs(np.median(ink_px, axis=0) - np.median(bg_px, axis=0))
    # Small text needs more contrast: its fine inter-stroke gaps vanish at low
    # contrast, so the floor rises as native height shrinks.
    thresh = 0.13 + max(0, 30 - native_h) * 0.006
    return float(d.max()) > thresh
