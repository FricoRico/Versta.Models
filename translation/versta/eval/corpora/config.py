"""
Steady-state configuration for the pinned FLORES-200 dev corpus snapshot.

The upstream constants pin *which* upstream release we snapshot; the
manifest in `evals/corpora/` pins *what we actually ship* (revision and
hashes). Changing either is a deliberate, reviewable edit.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class UpstreamSource:
    """
    Upstream dataset distribution a snapshot is built from.

    Attributes:
        name (str): Dataset name as cited in produced artifacts.
        url (str): Canonical download URL of the upstream archive.
        license (str): SPDX identifier of the dataset license.
        attribution (str): Attribution line carried into the artifact NOTICE.
    """

    name: str
    url: str
    license: str
    attribution: str


# FLORES-200 release tarball as published by Meta AI (No Language Left Behind
# project). The exact bytes are pinned by SHA-256 in the checked-in manifest;
# this URL only controls where a *new* snapshot revision is built from.
FLORES200_UPSTREAM = UpstreamSource(
    name="FLORES-200",
    url="https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    license="CC-BY-SA-4.0",
    attribution=(
        "NLLB Team, Marta R. Costa-jussa, James Cross, et al. "
        '"No Language Left Behind: Scaling Human-Centered Machine '
        'Translation" (2022), FLORES-200 evaluation dataset, Meta AI.'
    ),
)

# Sentence count of the FLORES-200 dev split. Every language file carries the
# same aligned block of sentences; a mismatch means the wrong split or a
# corrupted extraction, so builds fail fast against this constant.
FLORES200_DEV_SENTENCES = 997

# License notice packed into every snapshot artifact.
NOTICE_TEMPLATE = """{name} evaluation dataset, "{split}" split
Source:    {url}
License:   {license} (https://creativecommons.org/licenses/by-sa/4.0/)
Attribution:
{attribution}

This snapshot redistributes a subset of the upstream dataset (the languages
covering the Versta translation catalog) unmodified, under the terms of the
upstream license. If you modify it, share-alike obligations apply.
"""

# Canonical object-storage prefix for published corpus snapshots. The deploy
# step uploads `evals/corpora/<file>` here; the manifest records the full URL.
ARTIFACT_BASE_URL = "https://models.versta.app/evals/corpora"

# GitHub release that mirrors the artifact (tags are
# `evals-corpora-<name>-r<revision>`), so the snapshot is fetchable without
# object-storage access.
GITHUB_RELEASES_URL = "https://github.com/FricoRico/Versta.Models/releases/download"

# Catalog language code (as used in the checked-in `models.json`) ->
# FLORES-200 language code. Covers every language of the shipped catalog,
# including `mt` for the single-direction `Firefox/mt-lv` model the catalog
# bundles as `en-lv` (see `evals/corpora/README.md`).
FLORES200_LANGUAGE_MAP: Dict[str, str] = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "lv": "lvs_Latn",
    "mt": "mlt_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sv": "swe_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh-Hans": "zho_Hans",
}


def artifact_urls(name: str, revision: int) -> List[str]:
    """
    Mirror URLs of a published snapshot artifact, in fetch preference order.

    Args:
        name (str): Snapshot name (e.g. "flores200-dev").
        revision (int): Snapshot revision.

    Returns:
        list[str]: Canonical object-storage URL first, GitHub release mirror
        second. Both serve the same bytes; fetch verifies the SHA-256 from
        the manifest regardless of which mirror answered.
    """
    file_name = f"{name}-r{revision}.tar.gz"
    tag = f"evals-corpora-{name}-r{revision}"
    return [
        f"{ARTIFACT_BASE_URL}/{file_name}",
        f"{GITHUB_RELEASES_URL}/{tag}/{file_name}",
    ]
