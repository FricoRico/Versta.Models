"""Synthetic 48-px text strips with exact per-pixel ink/weight labels.

Generates (rgb strip, matte label, weight label, foreground/background colour
fields) for the glyph-matte U-Net. Text is rendered with RAqm-shaped Pillow
from a pinned, vendored font set (see `versta.train.glyphmatte.assets`) —
the font set and word lists are the reproducibility crux; host fontconfigs and
dictionaries shift under you.

Ported (approach, not files) from the MIT-licensed reference implementation by
David Ventura: translator-rs/scripts/ink_model/gen_data.py.

  https://github.com/DavidVentura/translator-rs/blob/master/scripts/ink_model/gen_data.py

Labels stay clean; only the composited RGB is degraded (see `synth.degrade`).
Annotated inspection sheets: `uv run python -m versta.dataset.glyphmatte.dump`.
"""

import random
import string
import unicodedata

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from .assets import assets_dir
from .config import (
    DISPLAY_HEAVY_FRAC,
    LATIN_DIGITS,
    SCRIPT_BLOCKS,
    SCRIPT_WEIGHTS,
    STRIP,
    WEIGHT,
    WORDS_BY_SCRIPT,
)
from .synth import coord_grid, degrade, legible, random_color


def _fonts(subdir: str, suffix: str) -> List[str]:
    d = assets_dir() / "fonts" / subdir
    files = sorted(str(p) for p in d.glob(f"*.{suffix}"))
    if not files:
        raise RuntimeError(
            f"no .{suffix} fonts under {d}; sync the pinned assets first with "
            "`uv run python -m versta.dataset.glyphmatte.assets`"
        )
    return files


@lru_cache(maxsize=1)
def latin_fonts() -> Tuple[str, ...]:
    return tuple(_fonts("latin", "ttf"))


@lru_cache(maxsize=1)
def display_fonts() -> Tuple[str, ...]:
    return tuple(_fonts("display", "ttf"))


@lru_cache(maxsize=1)
def cjk_fonts() -> Tuple[str, ...]:
    return tuple(_fonts("cjk", "ttc"))


@lru_cache(maxsize=1)
def shaped_fonts(script: str) -> Tuple[str, ...]:
    return tuple(_fonts(script, "ttf"))


def _in_blocks(token: str, blocks: Tuple[Tuple[int, int], ...]) -> bool:
    return all(any(lo <= ord(ch) <= hi for lo, hi in blocks) for ch in token)


@lru_cache(maxsize=1)
def english_words() -> Tuple[str, ...]:
    """English word source: vendored assets/words/en.txt; /usr/share/dict fallback."""
    src_ = assets_dir() / "words" / "en.txt"
    src = src_ if src_.exists() else Path("/usr/share/dict/words")
    words = [
        w.strip() for w in src.read_text().splitlines() if 2 <= len(w.strip()) <= 12
    ]
    return tuple(w for w in words if w.isalpha() or w == w.lower())


@lru_cache(maxsize=8)
def script_words(code: str) -> Tuple[str, ...]:
    """Top 5k tokens of a FrequencyWords list, filtered to the script's blocks.

    Subtitle-quality lists contain transliterations and mojibake; the block
    filter drops everything that isn't a real word of the target script.
    """
    path = assets_dir() / "words" / f"{code}.txt"
    if not path.exists():
        return ()
    out: List[str] = []
    for line in path.read_text().splitlines():
        token = line.split(" ", 1)[0].strip()
        if 2 <= len(token) <= 12 and _in_blocks(token, SCRIPT_BLOCKS[code]):
            out.append(token)
        if len(out) >= 5000:
            break
    return tuple(out)


def punctuate(rng: random.Random, text: str, script: str) -> str:
    """Adds light punctuation and digit runs to a word run.

    Strips without any punctuation teach the model to ignore punctuation
    strokes; these add it back at realistic rates.
    """
    if rng.random() < 0.12:
        # Numbers: 1-4 digits, occasionally with separators.
        digits = "".join(rng.choice(LATIN_DIGITS) for _ in range(rng.randint(1, 4)))
        insert = rng.random() < 0.5
        text = f"{text} {digits}" if insert else f"{digits} {text}"
    r = rng.random()
    if script == "cjk":
        if r < 0.15:
            return text + "。"
        if r < 0.22:
            return "「" + text + "」"
        if r < 0.3:
            return text + "、"
    else:
        if r < 0.30:
            return text + "."
        if r < 0.45:
            return text + ","
        if r < 0.55:
            return '"' + text + '"'
        if r < 0.6 and " " in text:
            # hyphenate two adjacent words: "foo bar" -> "foo-bar"
            a, b = text.split(" ", 1)
            return a + "-" + b
    return text


def latin_text(rng: random.Random) -> str:
    words = english_words()
    n = rng.randint(1, 6)
    parts = []
    for _ in range(n):
        if rng.random() < 0.85:
            w = rng.choice(words)
        else:
            w = "".join(
                rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8))
            )
        parts.append(w)
    text = " ".join(parts)
    r = rng.random()
    if r < 0.55:
        return text
    if r < 0.8:
        return text.title()
    return text.upper()


def script_text(rng: random.Random, script: str) -> str:
    """Picks words from the script's word list (random runs as fallback)."""
    if script == "latin":
        display = rng.random() < DISPLAY_HEAVY_FRAC
        return latin_text(rng).upper() if display else latin_text(rng)
    cand: List[str] = []
    for code in WORDS_BY_SCRIPT[script]:
        cand.extend(script_words(code))
    if not cand:
        # Fallback: random code points in the script's first block.
        lo, hi = SCRIPT_BLOCKS[WORDS_BY_SCRIPT[script][0]][0]
        cand = [chr(c) for c in range(lo, lo + 200)]
    if script == "cjk":
        return "".join(rng.choice(cand) for _ in range(rng.randint(2, 6)))
    if rng.random() < 0.15:
        # single repeated char run (keeps virama/combining-mark shapes present)
        lo, hi = SCRIPT_BLOCKS[WORDS_BY_SCRIPT[script][0]][0]
        # Filter to base letters only: combining marks are category M*.
        chars = [
            chr(c)
            for c in range(lo, lo + 128)
            if not unicodedata.category(chr(c)).startswith("M")
        ]
        return "".join(rng.choice(chars) for _ in range(rng.randint(3, 8)))
    n = rng.randint(1, 4)
    return " ".join(rng.choice(cand) for _ in range(n))


def pick_font(rng: random.Random, script: str) -> str:
    if script == "latin":
        if rng.random() < DISPLAY_HEAVY_FRAC:
            return rng.choice(display_fonts())
        return rng.choice(latin_fonts())
    if script == "cjk":
        return rng.choice(cjk_fonts())
    return rng.choice(shaped_fonts(script))


def render_mask(
    text: str,
    font_path: str,
    native_h: int,
    stroke_width: int,
) -> np.ndarray:
    """Renders the ink mask at SUPERSAMPLE× the strip geometry, H*SS rows.

    Returns the coverage mask (HxW at 3x) in 0..1. Canvas width fits the text.
    """
    ss = STRIP.supersample
    font = ImageFont.truetype(font_path, native_h * ss)
    probe = Image.new("L", (8, 8))
    d = ImageDraw.Draw(probe)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke_width * ss)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w = max(tw + 24 * ss, 8)
    h = STRIP.height * ss
    canvas = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    # Baseline placement: centre the glyph bbox vertically with jitter.
    x = -bbox[0] + 12 * ss
    y = (h - th) // 2 - bbox[1]
    d.text((x, y), text, font=font, fill=255, stroke_width=stroke_width * ss)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def stroke_ratio(mask3x: np.ndarray, native_h: int) -> float:
    """Effective stroke width as a fraction of glyph size, measured on the
    rendered mask: mean 2*EDT over glyph pixels (≈ mean local stroke width)
    over the glyph bbox height. Robust to display faces and synthetic strokes.
    """
    ink = mask3x > 0.5
    ys, xs = np.nonzero(ink)
    if len(ys) < 20:
        return 0.0
    edt = distance_transform_edt(ink)
    sw = 2.0 * float(edt[ink].mean())
    if sw <= 0:
        return 0.0
    bbox_h = max(float(ys.max() - ys.min() + 1), 1.0)
    return min(sw / bbox_h, 0.45)


def weight_target(ratio: float) -> float:
    """Continuous weight score 0..1 from the stroke-width ratio (logistic)."""
    t = (ratio - WEIGHT.mid) / WEIGHT.k
    t = max(min(t, 30.0), -30.0)
    return 1.0 / (1.0 + np.exp(-t))


def make_background(
    rng: random.Random, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Background RGB and the background colour field B (HxWx3 each, 0..1)."""
    bg = random_color(rng)
    if rng.random() < 0.5:
        bg = np.clip(0.6 + 0.4 * bg, 0, 1)  # bright paper-ish
    b_field = np.broadcast_to(bg, (h, w, 3)).astype(np.float32).copy()
    if rng.random() < 0.35:
        yy, xx = coord_grid(h, w)
        angle = rng.uniform(0, 2 * np.pi)
        t = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-16)
        g = rng.uniform(-0.25, 0.25)
        b_field = np.clip(b_field + (t[..., None] - 0.5) * g, 0, 1)
    img = b_field.copy()
    if rng.random() < 0.4:
        noise = np.random.default_rng(rng.getrandbits(32)).normal(
            0, rng.uniform(0.01, 0.05), (h, w, 1)
        )
        img = np.clip(img + noise, 1e-3, 1).astype(np.float32)
    return img, b_field


def ink_color(rng: random.Random, bg: np.ndarray) -> np.ndarray:
    """Foreground ink RGB; biased dark-on-light with occasional inversions."""
    if rng.random() < 0.08:
        return np.clip(random_color(rng) * 0.5 + 0.5, 0, 1)
    return np.clip(random_color(rng) * 0.45, 0, 1)


@dataclass
class Strip:
    """One labelled sample. matte/weight are HxW float32 in 0..1; rgb/fcol/
    bcol are HxWx3 float32 in 0..1."""

    rgb: np.ndarray
    matte: np.ndarray
    weight: np.ndarray
    fcol: np.ndarray
    bcol: np.ndarray
    params: Dict[str, object] = field(default_factory=dict)


def sample_strip(
    rng: random.Random,
    width: int = 320,
    max_tries: int = 12,
    keep_degrade: bool = True,
) -> Strip:
    """One labelled strip. Retries until the degraded image passes the
    legibility gate so the pair is actually learnable.

    Args:
        rng (random.Random): The sample RNG (worker-seeded deterministically).
        width (int): Target strip width in pixels (from WIDTHS).
        max_tries (int): Legibility retry budget.
        keep_degrade (bool): False for clean visualisations (skip degrade).

    Returns:
        Strip: The labelled strip.
    """
    for _ in range(max_tries):
        r = rng.random()
        acc = 0.0
        script = "latin"
        for s, w_ in SCRIPT_WEIGHTS.items():
            acc += w_ / sum(SCRIPT_WEIGHTS.values())
            if r < acc:
                script = s
                break
        text = punctuate(rng, script_text(rng, script), script)
        font_path = pick_font(rng, script)
        native_h = rng.randint(12, 44)
        stroke_w = rng.choices([0, 1, 2], weights=[86, 10, 4])[0]

        mask3x = render_mask(text, font_path, native_h, stroke_w)
        ss = STRIP.supersample
        if rng.random() < 0.4:
            img_m = Image.fromarray((mask3x * 255).astype(np.uint8))
            img_m = img_m.rotate(
                rng.uniform(-1.5, 1.5), resample=Image.BILINEAR, fillcolor=0
            )
            mask3x = np.asarray(img_m, dtype=np.float32) / 255.0

        m = Image.fromarray((mask3x * 255).astype(np.uint8)).resize(
            (max(mask3x.shape[1] // ss, 8), STRIP.height), Image.LANCZOS
        )
        cov = np.asarray(m, dtype=np.float32) / 255.0
        if cov.shape[1] >= width:
            x0 = rng.randint(0, cov.shape[1] - width)
            cov = cov[:, x0 : x0 + width]
        else:
            cov = np.pad(cov, ((0, 0), (0, width - cov.shape[1])))
        cov = np.ascontiguousarray(cov)

        ratio = stroke_ratio(mask3x, native_h)
        w_t = weight_target(ratio)
        matte = cov
        weight = (cov * w_t).astype(np.float32)

        bg_rgb, b_field = make_background(rng, STRIP.height, width)
        f = ink_color(rng, bg_rgb.mean(axis=(0, 1)))
        f_end: List[float] | None = None
        if rng.random() < 0.25:
            f2 = np.clip(f + np.array([rng.uniform(-0.3, 0.3) for _ in range(3)]), 0, 1)
            f_end = [round(float(c), 4) for c in f2]
            t = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
            f_map = np.broadcast_to(f, (STRIP.height, width, 3)).copy()
            f_map = (f_map * (1 - t) + f2 * t).astype(np.float32)
        else:
            f_map = (
                np.broadcast_to(f, (STRIP.height, width, 3)).astype(np.float32).copy()
            )

        a = cov[..., None]
        rgb = np.clip(a * f_map + (1 - a) * bg_rgb, 1e-3, 1).astype(np.float32)

        if keep_degrade:
            rgb = degrade(rgb, rng, native_h, photometric_aux=[f_map, b_field])
        if keep_degrade and not legible(rgb, matte, native_h):
            continue

        return Strip(
            rgb=rgb,
            matte=matte.astype(np.float32),
            weight=weight,
            fcol=f_map.astype(np.float32),
            bcol=b_field.astype(np.float32),
            params={
                "script": script,
                "font": Path(font_path).name,
                "native_h": native_h,
                "stroke_w": stroke_w,
                "stroke_ratio": round(ratio, 4),
                "weight_target": round(w_t, 3),
                "foreground_rgb": [round(float(c), 4) for c in f],
                "foreground_rgb_end": f_end,
                "background_rgb": [
                    round(float(c), 4) for c in bg_rgb.mean(axis=(0, 1))
                ],
                "text": text,
            },
        )
    raise RuntimeError("could not synthesise a legible strip in max_tries")
