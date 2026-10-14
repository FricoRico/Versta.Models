"""
Snapshot lifecycle: derive directions from the catalog, build the pinned
artifact from upstream, fetch it back, and verify extracted corpora.

Every step fails fast: a missing language, a wrong sentence count, a hash
mismatch or a malformed catalog aborts the run with a specific error rather
than producing or consuming an untrusted corpus.
"""

import json
import shutil

from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

from ...bundle.bundle_tar import create_checksum
from .archive import (
    extract_snapshot,
    pack_snapshot,
    read_tar_gz_members,
    sha256_file,
)
from .config import (
    FLORES200_DEV_SENTENCES,
    FLORES200_LANGUAGE_MAP,
    FLORES200_UPSTREAM,
    NOTICE_TEMPLATE,
    artifact_urls,
)
from .manifest import SCHEMA_ID, dump_manifest, load_manifest
from .typing import (
    CatalogEntry,
    CorpusManifest,
    ManifestDirection,
    ManifestFile,
    ManifestUpstream,
)

_DOWNLOAD_TIMEOUT_SECONDS = 120
_CHUNK_SIZE = 1024 * 1024


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def derive_directions(groups: List[List[CatalogEntry]]) -> List[Tuple[str, str, str]]:
    """
    Derive the evaluated directions from the checked-in model catalog.

    Bidirectional entries trust the catalog's `source_language`/`target_language`
    fields (and are checked for mirror symmetry within their group, the same
    invariant `versta.batch` enforces at export). Single-direction entries take
    the direction from `base_model`, which records the upstream model that was
    actually bundled — the catalog fields of the legacy `en-lv` group are
    unreliable there (its bundle was exported from `Firefox/mt-lv`).

    Args:
        groups (list[list[CatalogEntry]]): Catalog groups as loaded from
            `models.json` (a list of groups, each a list of entries).

    Returns:
        list[tuple[str, str, str]]: `(base_model, source, target)` per catalog
        entry, in catalog order.

    Raises:
        ValueError: If a bidirectional group is not mirrored, or a
            single-direction `base_model` cannot be split into two known
            language codes.
    """
    known = set(FLORES200_LANGUAGE_MAP)
    directions: List[Tuple[str, str, str]] = []

    for group in groups:
        field_directions = [
            (entry["source_language"], entry["target_language"]) for entry in group
        ]
        if all(entry["bidirectional"] for entry in group):
            mirrored = {(target, source) for source, target in field_directions}
            if set(field_directions) != mirrored:
                raise ValueError(
                    f"Bidirectional catalog group is not mirrored: {field_directions}"
                )
            directions.extend(
                (entry["base_model"], source, target)
                for entry, (source, target) in zip(group, field_directions)
            )
            continue

        for entry in group:
            identity = entry["base_model"].split("/", 1)[-1]
            splits = [
                (identity[:index], identity[index + 1 :])
                for index, char in enumerate(identity)
                if char == "-"
                and identity[:index] in known
                and identity[index + 1 :] in known
            ]
            if len(splits) != 1:
                raise ValueError(
                    f"Cannot resolve a direction for single-direction entry "
                    f"{entry['base_model']!r}: unambiguous split into catalog "
                    f"languages not found"
                )
            source, target = splits[0]
            if (source, target) != (
                entry["source_language"],
                entry["target_language"],
            ):
                print(
                    f"note: {entry['base_model']!r} catalog fields say "
                    f"{entry['source_language']}-{entry['target_language']}; "
                    f"using bundled model direction {source}-{target}"
                )
            directions.append((entry["base_model"], source, target))

    return directions


def build_snapshot(
    upstream_tarball: Path,
    catalog_path: Path,
    revision: int,
    output_dir: Path,
    manifest_path: Path,
) -> CorpusManifest:
    """
    Build a pinned corpus snapshot from the upstream dataset archive.

    Extracts the dev-split files for every language the catalog needs,
    repacks them (exact upstream bytes) with a license NOTICE and an embedded
    manifest into a deterministic `.tar.gz`, writes the checked-in manifest,
    and emits the artifact + a `.sha256` sidecar under `output_dir`.

    Rebuilding from the same upstream tarball, catalog and revision produces
    the same artifact SHA-256.

    Args:
        upstream_tarball (Path): The pinned upstream `flores200_dataset.tar.gz`.
        catalog_path (Path): The checked-in `models.json` catalog.
        revision (int): Snapshot revision; bump for any content change.
        output_dir (Path): Where the artifact and sidecar are written.
        manifest_path (Path): Where the checked-in manifest is written.

    Returns:
        CorpusManifest: The written manifest.

    Raises:
        ValueError: On missing languages, wrong sentence counts or an invalid
            catalog.
    """
    name = "flores200-dev"
    split = "dev"
    root = f"{name}-r{revision}"

    with open(catalog_path, encoding="utf-8") as handle:
        groups = cast(List[List[CatalogEntry]], json.load(handle))
    seeds = derive_directions(groups)

    used_codes = sorted(
        {code for _, source, target in seeds for code in (source, target)}
    )
    language_map = {code: FLORES200_LANGUAGE_MAP[code] for code in used_codes}

    prefix = "flores200_dataset/dev/"
    members = read_tar_gz_members(upstream_tarball, prefix)
    missing = [
        code for code in language_map.values() if f"{code}.{split}" not in members
    ]
    if missing:
        raise ValueError(
            f"{upstream_tarball} does not contain the dev split for: {missing}"
        )

    files: Dict[str, ManifestFile] = {}
    payloads: Dict[str, bytes] = {}
    for code in used_codes:
        flores_code = language_map[code]
        data = members[f"{flores_code}.{split}"]
        sentences = len(data.decode("utf-8").splitlines())
        if sentences != FLORES200_DEV_SENTENCES:
            raise ValueError(
                f"{prefix}{flores_code}.{split} has {sentences} sentences, "
                f"expected {FLORES200_DEV_SENTENCES}"
            )
        member = f"{split}/{flores_code}.{split}"
        files[member] = ManifestFile(sha256=_sha256_bytes(data), sentences=sentences)
        payloads[member] = data

    directions = [
        ManifestDirection(
            source=source,
            target=target,
            base_model=base_model,
            source_file=f"{split}/{language_map[source]}.{split}",
            target_file=f"{split}/{language_map[target]}.{split}",
            sentences=FLORES200_DEV_SENTENCES,
        )
        for base_model, source, target in seeds
    ]

    upstream = ManifestUpstream(
        name=FLORES200_UPSTREAM.name,
        url=FLORES200_UPSTREAM.url,
        sha256=sha256_file(upstream_tarball),
        size_bytes=upstream_tarball.stat().st_size,
        license=FLORES200_UPSTREAM.license,
        attribution=FLORES200_UPSTREAM.attribution,
    )

    embedded = {
        "schema": SCHEMA_ID,
        "name": name,
        "revision": revision,
        "split": split,
        "upstream": upstream,
        "files": files,
    }
    payloads["manifest.json"] = (json.dumps(embedded, indent=2) + "\n").encode("utf-8")
    payloads["NOTICE"] = NOTICE_TEMPLATE.format(
        name=upstream["name"],
        url=upstream["url"],
        license=upstream["license"],
        attribution=upstream["attribution"],
        split=split,
    ).encode("utf-8")

    artifact_file = f"{root}.tar.gz"
    artifact_path = pack_snapshot(payloads, root, Path(output_dir) / artifact_file)

    create_checksum(artifact_path)

    manifest = CorpusManifest(
        schema=SCHEMA_ID,
        name=name,
        revision=revision,
        split=split,
        upstream=upstream,
        artifact={
            "file": artifact_file,
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "urls": artifact_urls(name, revision),
        },
        language_map=language_map,
        files=files,
        directions=directions,
    )
    dump_manifest(manifest, manifest_path)
    print(
        f"Built {artifact_file}: {len(files)} languages, {len(directions)} "
        f"directions, {FLORES200_DEV_SENTENCES} sentences/direction; "
        f"manifest written to {manifest_path}"
    )
    return manifest


def fetch_snapshot(manifest_path: Path, output_dir: Path, check: bool) -> Path:
    """
    Fetch the pinned snapshot recorded in the manifest and verify it fully.

    Same manifest in, byte-identical corpus out: the artifact is checked
    against the recorded SHA-256 before extraction, and every extracted file
    is checked against the recorded per-file hashes and sentence counts.

    Args:
        manifest_path (Path): The checked-in manifest.
        output_dir (Path): Cache directory; the corpus lands in
            `<output_dir>/<name>-r<revision>/`.
        check (bool): Only verify an already-fetched corpus, never download.

    Returns:
        Path: The verified corpus directory containing `dev/`.

    Raises:
        ValueError: On hash/count mismatches, or when no recorded URL serves
            the artifact.
    """
    manifest = load_manifest(manifest_path)
    corpus_dir = Path(output_dir) / f"{manifest['name']}-r{manifest['revision']}"

    if corpus_dir.exists():
        try:
            verify_corpus(corpus_dir, manifest)
            print(f"Corpus already present and verified: {corpus_dir}")
            return corpus_dir
        except ValueError as error:
            if check:
                raise
            print(f"Discarding corrupted corpus at {corpus_dir} ({error})")
            shutil.rmtree(corpus_dir)
    elif check:
        raise ValueError(f"--check failed: no corpus at {corpus_dir}")

    artifact_path = _download_artifact(manifest, Path(output_dir))
    extracted = extract_snapshot(artifact_path, Path(output_dir))
    if extracted != corpus_dir:
        raise ValueError(
            f"Artifact unpacked to {extracted.name!r}, expected {corpus_dir.name!r}"
        )
    verify_corpus(corpus_dir, manifest)
    print(
        f"Corpus verified: {corpus_dir} "
        f"({len(manifest['files'])} languages, "
        f"{len(manifest['directions'])} directions)"
    )
    return corpus_dir


def fetch_upstream(upstream_url: str, output_dir: Path) -> Path:
    """
    Download the upstream dataset archive for a snapshot build.

    The observed SHA-256 is recorded in the built manifest; building a pinned
    revision twice must come out byte-identical, so prefer a local
    `--upstream_tarball` when regenerating a released revision.

    Args:
        upstream_url (str): URL of the upstream archive.
        output_dir (Path): Directory the archive is downloaded into.

    Returns:
        Path: Local path of the downloaded archive.
    """
    local = Path(output_dir) / upstream_url.rsplit("/", 1)[-1]
    if local.is_file():
        print(f"Upstream archive already present: {local}")
        return local

    print(f"Downloading upstream dataset: {upstream_url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        urlopen(
            Request(upstream_url, method="GET"), timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response,
        open(local, "wb") as handle,
    ):
        while block := response.read(_CHUNK_SIZE):
            handle.write(block)
    print(f"Upstream archive sha256: {sha256_file(local)}")
    return local


def _download_artifact(manifest: CorpusManifest, output_dir: Path) -> Path:
    """
    Download the snapshot artifact from its recorded mirrors, hash-verified.

    Args:
        manifest (CorpusManifest): The loaded manifest.
        output_dir (Path): Directory the artifact is downloaded into.

    Returns:
        Path: Local path of the artifact matching the recorded SHA-256.

    Raises:
        ValueError: If every mirror fails or serves wrong bytes.
    """
    artifact = manifest["artifact"]
    local = Path(output_dir) / artifact["file"]
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    for url in artifact["urls"]:
        try:
            print(f"Downloading {url}")
            digest = sha256()
            with (
                urlopen(
                    Request(url, method="GET"), timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                open(local, "wb") as handle,
            ):
                while block := response.read(_CHUNK_SIZE):
                    digest.update(block)
                    handle.write(block)
            if digest.hexdigest() != artifact["sha256"]:
                errors.append(f"{url}: sha256 mismatch")
                continue
            return local
        except (OSError, URLError) as error:
            errors.append(f"{url}: {error}")

    raise ValueError(
        f"Could not fetch {artifact['file']} from any mirror: " + "; ".join(errors)
    )


def verify_corpus(corpus_dir: Path, manifest: CorpusManifest) -> None:
    """
    Verify an extracted corpus against the manifest, file by file.

    Args:
        corpus_dir (Path): Snapshot root containing `dev/`.
        manifest (CorpusManifest): A validated manifest (via `load_manifest`).

    Raises:
        ValueError: Listing every file that is missing, has a wrong hash or a
            wrong sentence count.
    """
    problems: List[str] = []
    for member, entry in manifest["files"].items():
        path = Path(corpus_dir) / member
        if not path.is_file():
            problems.append(f"{member}: missing")
            continue
        data = path.read_bytes()
        if _sha256_bytes(data) != entry["sha256"]:
            problems.append(f"{member}: sha256 mismatch")
            continue
        sentences = len(data.decode("utf-8").splitlines())
        if sentences != entry["sentences"]:
            problems.append(
                f"{member}: {sentences} sentences, expected {entry['sentences']}"
            )
    if problems:
        raise ValueError(
            f"Corpus at {corpus_dir} failed verification: " + "; ".join(problems)
        )
