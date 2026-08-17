"""Tunables for the glyphmatte synthetic strip dataset.

Everything a generator/parquet/materializer run can be steered with lives
here, grouped per theme so the numeric annotations stay next to the numbers
they describe. Callers import the instances, not the classes.
"""

import os

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class StripGeometry:
    """Strip target geometry: the app dewarps lines to `height` px tall and
    pads each width bucket; `widths` is the runtime bucket list."""

    height: int = 48
    widths: List[int] = field(default_factory=lambda: list(range(96, 512 + 1, 16)))
    supersample: int = 3


STRIP = StripGeometry()


@dataclass(frozen=True)
class WeightCurve:
    """Logistic weight target over the measured stroke-width ratio (stroke
    width as fraction of the glyph bbox height, measured on the rendered
    mask). Calibrated against the vendored Noto set at 48px: regular ≈0.06 →
    ~0.1, bold ≈0.09 → ~0.9, heavy display faces 0.10–0.25 → ~1."""

    mid: float = 0.075
    k: float = 0.008


WEIGHT = WeightCurve()


@dataclass(frozen=True)
class DatasetConfig:
    """Materialisation defaults (counts and shard sizing)."""

    n: int = 50000
    val_n: int = 1024
    shard_size: int = 5000
    seed: int = 1787167204


DATASET = DatasetConfig()

# Fraction of latin samples forced through the display pool (heavy condensed
# faces for header-style text the regular pool never reaches). Overridable via
# environment for one-off sampling experiments.
DISPLAY_HEAVY_FRAC = float(os.environ.get("GLYPHMATTE_DISPLAY_HEAVY", "0.02"))

LATIN_DIGITS = "0123456789"

# Script code -> (unicode block start, end) pairs accepted in word lists. Keeps
# transliterated latin / mojibake (broken UTF-8 sequences decode as long runs
# of one block too — those entries mix in latin or invalid chars and get
# dropped by the all-chars-in-block filter).
SCRIPT_BLOCKS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "ar": ((0x0600, 0x06FF),),
    "hi": ((0x0900, 0x097F),),
    "ta": ((0x0B80, 0x0BFF),),
    "th": ((0x0E00, 0x0E7F),),
    "zh_cn": ((0x4E00, 0x9FFF),),
    "ja": ((0x4E00, 0x9FFF), (0x3040, 0x30FF)),
    # ko is mostly Hangul but allows Han.
    "ko": ((0xAC00, 0xD7A3), (0x4E00, 0x9FFF)),
}

# Word-list groupings per generator script: zh_CN is all Han; ja is Han+kana;
# ko is Hangul.
WORDS_BY_SCRIPT: Dict[str, Tuple[str, ...]] = {
    "arabic": ("ar",),
    "devanagari": ("hi",),
    "tamil": ("ta",),
    "thai": ("th",),
    "cjk": ("zh_cn", "ja", "ko"),
}

# Sampling mix over generator scripts.
SCRIPT_WEIGHTS: Dict[str, int] = {
    "latin": 50,
    "cjk": 20,
    "arabic": 10,
    "devanagari": 8,
    "tamil": 6,
    "thai": 6,
}

PARQUET_COLUMNS = ("rgb", "matte", "weight", "foreground", "background")
