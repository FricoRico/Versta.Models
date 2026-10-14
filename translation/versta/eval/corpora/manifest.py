"""
Load, validate and serialize the checked-in corpus manifest.

The manifest is the contract between the snapshot builder and every consumer
(CI, the eval harness, a developer's machine). Validation is fail-fast: a
malformed manifest is a build bug, not a runtime condition to tolerate.
"""

import json
import re

from pathlib import Path
from typing import Any, Callable, NoReturn, cast

from .typing import CorpusManifest

SCHEMA_ID = "versta/corpus-manifest@1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: Path) -> CorpusManifest:
    """
    Load and validate a corpus manifest.

    Args:
        path (Path): Path of the checked-in manifest JSON.

    Returns:
        CorpusManifest: The validated manifest.

    Raises:
        ValueError: If the manifest is missing fields, has malformed hashes
            or is internally inconsistent.
    """
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    manifest = cast(CorpusManifest, document)
    validate_manifest(manifest, source=path)
    return manifest


def dump_manifest(manifest: CorpusManifest, path: Path) -> Path:
    """
    Serialize a manifest to its checked-in form (stable key order, 2-space
    indent, trailing newline) so diffs stay minimal and reviewable.

    Args:
        manifest (CorpusManifest): The manifest to write.
        path (Path): Destination path.

    Returns:
        Path: `path`, for chaining.
    """
    validate_manifest(manifest, source=path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return path


def validate_manifest(manifest: CorpusManifest, source: Path) -> None:
    """
    Fail-fast structural and consistency checks on a manifest.

    Args:
        manifest (CorpusManifest): The manifest to validate.
        source (Path): Where the manifest came from, used in error messages.

    Raises:
        ValueError: On any schema violation, malformed hash or inconsistency
            between `directions`, `files` and `language_map`.
    """

    def fail(reason: str) -> NoReturn:
        raise ValueError(f"Invalid corpus manifest {source}: {reason}")

    if manifest.get("schema") != SCHEMA_ID:
        fail(f"schema must be {SCHEMA_ID!r}")
    if not manifest.get("name"):
        fail("missing name")
    if not manifest.get("split"):
        fail("missing split")
    revision = manifest.get("revision")
    if not isinstance(revision, int) or revision < 1:
        fail("revision must be a positive integer")

    upstream = manifest.get("upstream")
    artifact = manifest.get("artifact")
    if not upstream or "url" not in upstream:
        fail("missing upstream.url")
    _require_hash(upstream.get("sha256"), "upstream.sha256", fail)
    if not upstream.get("license") or not upstream.get("attribution"):
        fail("upstream.license and upstream.attribution are required")

    if not artifact or not artifact.get("file"):
        fail("missing artifact.file")
    _require_hash(artifact.get("sha256"), "artifact.sha256", fail)
    if not artifact.get("urls"):
        fail("artifact.urls must list at least one mirror")

    files = manifest.get("files") or {}
    if not files:
        fail("files must not be empty")
    for name, entry in files.items():
        _require_hash(entry.get("sha256"), f"files[{name!r}].sha256", fail)
        sentences = entry.get("sentences")
        if not isinstance(sentences, int) or sentences < 1:
            fail(f"files[{name!r}].sentences must be a positive integer")

    language_map = manifest.get("language_map") or {}
    if not language_map:
        fail("language_map must not be empty")

    directions = manifest.get("directions") or []
    if not directions:
        fail("directions must not be empty")
    for direction in directions:
        label = f"{direction.get('source')}-{direction.get('target')}"
        for code_key, code in (
            ("source", direction.get("source")),
            ("target", direction.get("target")),
        ):
            if not isinstance(code, str) or code not in language_map:
                fail(
                    f"direction {label} uses {code_key} {code!r} missing from "
                    f"language_map"
                )
        for file_key, member in (
            ("source_file", direction.get("source_file")),
            ("target_file", direction.get("target_file")),
        ):
            if not isinstance(member, str) or member not in files:
                fail(f"direction {label} references unknown file {member!r}")
            if files[member]["sentences"] != direction.get("sentences"):
                fail(f"direction {label} sentence count does not match {member!r}")

    split = manifest["split"]
    expected_files = {f"{split}/{code}.{split}" for code in language_map.values()}
    if set(files) != expected_files:
        fail(
            f"files and language_map disagree: only in files: "
            f"{sorted(set(files) - expected_files)}, only in language_map: "
            f"{sorted(expected_files - set(files))}"
        )


def _require_hash(value: Any, field: str, fail: Callable[[str], NoReturn]) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        fail(f"{field} must be a lowercase sha256 hex digest")
