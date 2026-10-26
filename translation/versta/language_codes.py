"""
The language-code contract shared by the catalog, the bundles and the app
(NEU-133).

There are exactly two code layers, and this module owns the mapping between
them:

  * **Catalog layer** (checked-in ``models.json``): BCP-47-style codes as the
    app presents them, e.g. ``zh-Hans`` for simplified Chinese. These are
    descriptive fields owned by the catalog.
  * **Bundle-boundary layer** (direction ``metadata.json``, bundle-level
    ``metadata.json.languages``, ``<slug>-bundle.tar.gz`` archive names):
    Mozilla registry pair keys — bare ISO-639 codes, with Chinese kept as
    ``zh`` (simplified) / ``zh_hant`` (traditional). This layer is canonical
    at the artifact boundary: published bundles stay byte-aligned with the
    upstream registry payloads, and archive names stay derivable from bundle
    contents.

``normalize_language`` is the single, only mapping between the layers. The
app's install-time join of catalog entries to bundle-declared languages MUST
compare codes through this normalization on both sides — never string-equal
raw catalog codes against raw bundle codes (``zh-Hans`` never equals ``zh``).
``versta.batch``'s catalog sync already keys on the normalized pair
(``_entry_key``), and the schema gate enforces that every catalog entry's
bundle URL basename is the deterministic slug of its normalized languages, so
the two layers cannot silently drift apart.
"""

import re

from typing import Iterable, List


def normalize_language(code: str) -> str:
    """
    Normalizes a user-supplied language code to the form used as a key in the Firefox registry.

    The registry uses bare ISO-639 codes (e.g. "en", "nl") and, for Chinese, distinguishes
    "zh" (simplified) from "zh_hant" (traditional). Script- or region-qualified inputs are mapped
    accordingly so callers may request models using common BCP-47-style tags:

      * "zh-Hans"/"zh-CN"/"zh"            -> "zh"      (simplified)
      * "zh-Hant"/"zh-TW"/"zh-HK"/"zh-MO" -> "zh_hant" (traditional)
      * "en-US"/"en-GB"/"pt-BR"          -> "en"/"pt" (region subtag stripped)
      * "no"/"nb"/"nn"/"hbs"             -> unchanged (distinct real pairs, never collapsed)

    Args:
        code (str): A language code, possibly with script/region subtags (e.g. "zh-Hans").

    Returns:
        str: The normalized language code used in registry pair keys.
    """
    parts = re.split(r"[-_]", code.lower())
    base = parts[0]

    if base == "zh":
        return (
            "zh_hant"
            if any(part in ("hant", "tw", "hk", "mo") for part in parts[1:])
            else "zh"
        )

    return base


def bundle_languages(codes: Iterable[str]) -> List[str]:
    """
    Canonical language list for a bundle covering ``codes``: normalized and
    sorted. This is both the bundle-level ``metadata.json.languages`` value
    and the ordering the archive name is derived from.

    Args:
        codes (Iterable[str]): Direction language codes going into one bundle.

    Returns:
        List[str]: Sorted, de-duplicated normalized codes.
    """
    return sorted({normalize_language(code) for code in codes})


def bundle_slug(codes: Iterable[str]) -> str:
    """
    Deterministic bundle identity for a set of direction codes: the stem of
    ``<slug>-bundle.tar.gz`` exactly as ``versta.bundle`` names archives
    (e.g. {"zh-Hans", "en"} -> "en-zh").

    Args:
        codes (Iterable[str]): Direction language codes going into one bundle.

    Returns:
        str: The bundle slug built from the normalized, sorted codes.
    """
    return "-".join(bundle_languages(codes))
