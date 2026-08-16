import json

from hashlib import sha256
from pathlib import Path
from typing import List, Optional, cast

from .definitions import PACK_NAME
from .typing import Manifest, ManifestFile, ManifestRole


def _sha256(path: Path) -> str:
    """
    Computes the hex SHA256 of a file, streaming in 1 MiB chunks.

    Args:
        path (Path): File to hash.

    Returns:
        str: The lowercase hex digest.
    """
    digest = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(
    path: Path,
    role: ManifestRole,
    priority: int,
    script: Optional[str] = None,
    note: Optional[str] = None,
) -> ManifestFile:
    """
    Builds a manifest entry for a produced pack file.

    Args:
        path (Path): The produced file inside the pack directory.
        role (ManifestRole): The file's role in the OCR pipeline.
        priority (int): Selection priority within the role (1 = preferred).
        script (Optional[str]): Script slot for recognizers and keys
            ("latin"/"cj").
        note (Optional[str]): Free-form usage note (e.g. stills-only).

    Returns:
        ManifestFile: The manifest entry.
    """
    entry = ManifestFile(
        name=path.name,
        sizeBytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
        priority=priority,
    )
    if script is not None:
        entry["script"] = script
    if note is not None:
        entry["note"] = note
    return entry


def write_manifest(version: str, pack_dir: Path, files: List[ManifestFile]) -> Path:
    """
    Writes the pack manifest (`manifest.json`) listing every file the app
    downloads, with sizes, checksums, roles and priorities.

    Entries produced by this run replace same-named entries already in the
    manifest, so partial re-runs (`--models`) keep the rest of the pack
    intact.

    Args:
        version (str): Pack version stamp (from versta/version.txt).
        pack_dir (Path): The pack output directory.
        files (List[ManifestFile]): Manifest entries for the produced files.

    Returns:
        Path: The written manifest path.
    """
    manifest_path = pack_dir / "manifest.json"
    merged: List[ManifestFile] = []
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            merged = cast(Manifest, json.load(f))["files"]
    for entry in files:
        merged = [e for e in merged if e["name"] != entry["name"]] + [entry]

    manifest = Manifest(version=version, pack=PACK_NAME, files=merged)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    return manifest_path
