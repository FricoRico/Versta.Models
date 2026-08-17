"""Pinned font and word-list downloads for the synthetic data generator.

Fonts are the reproducibility crux of the glyphmatte training data: host
install sets shift under you, so a fixed set of Noto/display families is
vendored under `assets/fonts/` (gitignored) with a sha256 pin per file.

Word lists come from hermitdave/FrequencyWords (MIT, subtitle frequency data)
filtered per script by Unicode block at load time; English uses the dwyl list.
Both live under `assets/words/`, sha256-pinned like the fonts.

Fonts: SIL OFL (Noto families); display faces (Anton, BebasNeue, Staatliches,
AlfaSlabOne) are OFL. No proprietary fonts.

CLI: uv run python -m versta.train.glyphmatte.assets
"""

import hashlib
import os

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from urllib.request import Request, urlopen

from tqdm import tqdm

MODULE_ROOT = Path(__file__).parents[3]

# Assets live under <output_dir>/intermediates/assets by default (intermediates
# are deleted unless --keep_intermediates). Spawn workers inherit the env var,
# which is why the directory travels through the environment rather than args.
_DEFAULT_ASSETS = "cache/dataset/glyphmatte"
ASSETS_ENV = "GLYPHMATTE_ASSETS"


def assets_dir() -> Path:
    return Path(os.environ.get(ASSETS_ENV, _DEFAULT_ASSETS))


def set_assets_dir(path: Path) -> None:
    os.environ[ASSETS_ENV] = str(path)


# The directory may change at runtime via set_assets_dir (the pipeline points
# it at <output_dir>/intermediates/assets), so every call resolves it fresh
# through assets_dir(); nothing is captured at import time.


class Asset(NamedTuple):
    """A pinned downloadable file.

    url: upstream URL; dest: path relative to assets_dir(); sha256: hex digest or
    None to compute+report without failing (bootstrapping).
    """

    url: str
    dest: str
    sha256: Optional[str] = None


GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts/main/ofl"
NOTO_FONTS = (
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts"
)
NOTO_CJK = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans"
FREQ_WORDS = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"
)


def _noto(family: str, style: str) -> str:
    return f"{NOTO_FONTS}/{family}/hinted/ttf/{family}-{style}.ttf"


FONTS: List[Asset] = [
    Asset(
        _noto("NotoSans", "Regular"),
        "fonts/latin/NotoSans-Regular.ttf",
        "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823",
    ),
    Asset(
        _noto("NotoSans", "Bold"),
        "fonts/latin/NotoSans-Bold.ttf",
        "1df075a380fc7cb898acf64c1f7b3b4dd780de3caa860178bf929de35817a913",
    ),
    Asset(
        _noto("NotoSerif", "Regular"),
        "fonts/latin/NotoSerif-Regular.ttf",
        "19e72cd8d595fae5bd74a5206f5d938512e1183d4fed7abb1ec1be1d7efa5f88",
    ),
    Asset(
        _noto("NotoSerif", "Bold"),
        "fonts/latin/NotoSerif-Bold.ttf",
        "96656aa5cec8f1d6fd0e804c1fad397e1a1cfa082e6642124e0bda68cd8363ce",
    ),
    Asset(
        _noto("NotoSansMono", "Regular"),
        "fonts/latin/NotoSansMono-Regular.ttf",
        "65b5e2b2c4a1fba9ae8be1f026cb35b03dcb8886d9b2a4147054fde12f7e767d",
    ),
    Asset(
        _noto("NotoSansMono", "Bold"),
        "fonts/latin/NotoSansMono-Bold.ttf",
        "a21ea0ba6ea49fda7b34ca39a504b487f1130885d36e1a4f9f4255b3ba6994bc",
    ),
    Asset(
        f"{GOOGLE_FONTS}/anton/Anton-Regular.ttf",
        "fonts/display/Anton-Regular.ttf",
        "a4ba3a92350ebb031da0cb47630ac49eb265082ca1bc0450442f4a83ab947cab",
    ),
    Asset(
        f"{GOOGLE_FONTS}/bebasneue/BebasNeue-Regular.ttf",
        "fonts/display/BebasNeue-Regular.ttf",
        "08e4623805102d819f58601e46e345648846075e363b2ceb23313c2d1c83ec73",
    ),
    Asset(
        f"{GOOGLE_FONTS}/staatliches/Staatliches-Regular.ttf",
        "fonts/display/Staatliches-Regular.ttf",
        "8395212aa4c6c3534bd39a745d956305ff080c3f3ed73ba61e4fbaae951e55cc",
    ),
    Asset(
        f"{GOOGLE_FONTS}/alfaslabone/AlfaSlabOne-Regular.ttf",
        "fonts/display/AlfaSlabOne-Regular.ttf",
        "28664afa698a3393bd5a29eec750230a0645c5301d62200e5f2d3a027fb2299d",
    ),
    Asset(
        f"{NOTO_CJK}/OTC/NotoSansCJK-Regular.ttc",
        "fonts/cjk/NotoSansCJK-Regular.ttc",
        "b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a",
    ),
    Asset(
        f"{NOTO_CJK}/OTC/NotoSansCJK-Bold.ttc",
        "fonts/cjk/NotoSansCJK-Bold.ttc",
        "faa5f3656a78b2e2d450d27fe8382c778bc2b6bb5ea29c986664a6a435056ceb",
    ),
    Asset(
        _noto("NotoSansArabic", "Regular"),
        "fonts/arabic/NotoSansArabic-Regular.ttf",
        "bdff3e5659d67e67def05b33f749683b9376ae819d65d3dd62ac4640b3aaef48",
    ),
    Asset(
        _noto("NotoSansArabic", "Bold"),
        "fonts/arabic/NotoSansArabic-Bold.ttf",
        "4e5462d2e8be880317b9f49b5b2da109ddb6a3563d91cc604b67f3535832a555",
    ),
    Asset(
        _noto("NotoSansDevanagari", "Regular"),
        "fonts/devanagari/NotoSansDevanagari-Regular.ttf",
        "306b53ecfb182a504dd8a7446093c316387d2fd8dc350d0792ed1753fe0996cd",
    ),
    Asset(
        _noto("NotoSansDevanagari", "Bold"),
        "fonts/devanagari/NotoSansDevanagari-Bold.ttf",
        "3ad8362a06271814869838dcc3d161b13c9fb97681b627af1f7f283ea9387d56",
    ),
    Asset(
        _noto("NotoSansTamil", "Regular"),
        "fonts/tamil/NotoSansTamil-Regular.ttf",
        "3c0a186feb3c63c7f6d63e1511dcdc144e745ae09b98e217c83f3e317974f6f9",
    ),
    Asset(
        _noto("NotoSansTamil", "Bold"),
        "fonts/tamil/NotoSansTamil-Bold.ttf",
        "683682d585698b8b44da066d4903762d8aaa471bdffb919ac869753689ba3950",
    ),
    Asset(
        _noto("NotoSansThai", "Regular"),
        "fonts/thai/NotoSansThai-Regular.ttf",
        "61cf814eec46b294d6ea4401ac295d0cecd5207bd2331dcc5a15e7301d30ee44",
    ),
    Asset(
        _noto("NotoSansThai", "Bold"),
        "fonts/thai/NotoSansThai-Bold.ttf",
        "2ac6c6e8a478e23b15f76e4894af1fa2210f8f350e4e6e54aad530bec03efbfb",
    ),
]

_EN_WORDS = (
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
)

WORDS: List[Asset] = [
    Asset(
        _EN_WORDS,
        "words/en.txt",
        "3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48",
    ),
    *[
        Asset(f"{FREQ_WORDS}/{c}/{c}_50k.txt", f"words/{c}.txt", h)
        for c, h in (
            ("ar", "bbe98b4b92902b392bdefa2e555a108fdb42a5dd79d261674be5ab666229e19f"),
            ("th", "20e7052f2d64222e1420c5d0b4ed6b68cd6290f0cf8b908d8bc6b0af781b6083"),
            ("ko", "8d00401b6728c8d6feeaa5455583c1a33408dd020aae4aa55609b25004d1d99d"),
            (
                "zh_cn",
                "25599e00347b893e55c3058a748819b3fe19403291cd12f7c03af128e7f3fe45",
            ),
        )
    ],
    # hi/ta/ja lack 50k files — the 2018 *_full.txt is the frequency list.
    *[
        Asset(f"{FREQ_WORDS}/{c}/{c}_full.txt", f"words/{c}.txt", h)
        for c, h in (
            ("hi", "7ea1238ba983a3aae456c73d14e5f031ec59a1316d665af9b120d0c0fee61a8c"),
            ("ta", "62105bea7448f5243dc4bfbed5311dd607f08af21fa026c776d2a373c10a24f8"),
            ("ja", "581dd106a134ff9f2706e23cde9fe95cd24d96ead6357cd3bd3bba9e166a764e"),
        )
    ],
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(asset: Asset) -> Tuple[Path, str]:
    dest = assets_dir() / asset.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(asset.url, method="GET"), timeout=600) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with open(dest, "wb") as out:
            with tqdm(
                desc=dest.name,
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    bar.update(len(chunk))
    return dest, _sha256(dest)


def missing_assets() -> List[Asset]:
    """Pinned assets not yet present under the current assets dir."""
    return [a for a in FONTS + WORDS if not (assets_dir() / a.dest).exists()]


def sync_assets() -> Dict[str, str]:
    """Downloads every pinned asset; prints the sha256 of each so pins can be
    filled in. Skips files that already exist.

    Returns:
        Dict[str, str]: dest path -> sha256 for every asset.
    """
    out: Dict[str, str] = {}
    for asset in FONTS + WORDS:
        dest = assets_dir() / asset.dest
        if dest.exists():
            out[asset.dest] = _sha256(dest)
            continue
        dest, digest = _download(asset)
        out[asset.dest] = digest
        if asset.sha256 and digest != asset.sha256:
            raise RuntimeError(f"{asset.dest}: sha256 mismatch")
    return out


def parse_args() -> Namespace:
    return ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = sync_assets()
    for dest, digest in result.items():
        print(f"{dest}  {digest}")


if __name__ == "__main__":
    main()
