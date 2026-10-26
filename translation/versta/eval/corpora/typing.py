"""Typed shapes of the checked-in corpus manifest and the catalogs it reads."""

from typing import Dict, List, TypedDict


class ManifestUpstream(TypedDict):
    """
    Provenance of the upstream dataset a snapshot was built from.

    Attributes:
        name (str): Dataset name (e.g. "FLORES-200").
        url (str): URL the upstream archive was fetched from.
        sha256 (str): SHA-256 of the exact upstream archive bytes.
        size_bytes (int): Size of the upstream archive.
        license (str): SPDX identifier of the dataset license.
        attribution (str): Attribution required by the dataset license.
    """

    name: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    attribution: str


class ManifestArtifact(TypedDict):
    """
    The published snapshot artifact, pinned by content.

    Attributes:
        file (str): Artifact file name (e.g. "flores200-dev-r1.tar.gz").
        sha256 (str): SHA-256 of the artifact; fetch fails closed on mismatch.
        size_bytes (int): Size of the artifact.
        urls (list[str]): Mirror URLs in fetch preference order.
    """

    file: str
    sha256: str
    size_bytes: int
    urls: List[str]


class ManifestFile(TypedDict):
    """
    One corpus file inside the snapshot artifact.

    Attributes:
        sha256 (str): SHA-256 of the extracted file bytes.
        sentences (int): Number of sentences (lines) in the file.
    """

    sha256: str
    sentences: int


class ManifestDirection(TypedDict):
    """
    One evaluated translation direction covered by the snapshot.

    Attributes:
        source (str): Catalog source language code.
        target (str): Catalog target language code.
        base_model (str): Upstream model identity the direction came from.
        source_file (str): Archive path of the source-side sentences.
        target_file (str): Archive path of the reference translations.
        sentences (int): Aligned sentence count for the direction.
    """

    source: str
    target: str
    base_model: str
    source_file: str
    target_file: str
    sentences: int


class CorpusManifest(TypedDict):
    """
    The checked-in corpus snapshot manifest (`evals/corpora/flores200-dev.json`).

    Same manifest in, byte-identical corpus out: `files` and `artifact` pin
    every byte of the fetched corpus by SHA-256.
    """

    schema: str
    name: str
    revision: int
    split: str
    upstream: ManifestUpstream
    artifact: ManifestArtifact
    language_map: Dict[str, str]
    files: Dict[str, ManifestFile]
    directions: List[ManifestDirection]


class CatalogEntry(TypedDict):
    """
    The fields of one `models.json` catalog entry the corpus build reads.

    Attributes:
        base_model (str): Upstream model identity (e.g. "Firefox/en-es").
        source_language (str): Catalog source language code.
        target_language (str): Catalog target language code.
        bidirectional (bool): Whether the group covers both directions.
    """

    base_model: str
    source_language: str
    target_language: str
    bidirectional: bool
