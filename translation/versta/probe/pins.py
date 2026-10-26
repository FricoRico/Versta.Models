"""
Verification of downloaded model directions against the pack-acceptance
content pins.

The acceptance harness downloads real Firefox translations in CI. Mozilla
moves the registry underneath us (new exports, re-uploaded vocabularies), so
pack-acceptance pins the *content* SHA-256 of every artifact direction it
probes in ``pack-acceptance/pins.json`` — same re-baselining discipline as
the leanmt bench baselines: a drift fails CI deliberately, and the pins move
only in a dedicated commit with the measured scorecard diff attached.
"""

import json

from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

from .install import sha256_of


class DirectionPins(TypedDict):
    """Pinned content hashes for one downloaded direction."""

    files: Dict[str, str]


class PinsError(ValueError):
    """A downloaded file's content diverged from its pin."""


def load_published_pins(pins_path: Path) -> Dict[str, Any]:
    """
    Loads the ``published`` section: tar checksums for the served bundles plus
    the per-file payload pins measured on them at first-sweep time. Used by the
    full-catalog sweep (NEU-59) so a silent re-publish of models.versta.app is
    caught deliberately, same discipline as registry drift.

    Args:
        pins_path (Path): path to pack-acceptance/pins.json.

    Returns:
        Dict[str, Any]: the raw ``published`` document.

    Raises:
        PinsError: on a malformed document.
    """
    with open(pins_path, "r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    published: Any = data.get("published")
    if not isinstance(published, dict):
        raise PinsError(f"{pins_path}: missing 'published' section")
    for required in ("bundle_base_url", "directions", "bundles"):
        if required not in published:
            raise PinsError(f"{pins_path}: 'published' lacks {required!r}")
    return cast(Dict[str, Any], published)


def verify_published_bundle(bundle_path: Path, expected_sha256: str) -> None:
    """
    Verifies one served bundle tarball against its pinned sha256.

    Args:
        bundle_path (Path): the downloaded ``<slug>-bundle.tar.gz``.
        expected_sha256 (str): the pin from pins.json ``published.bundles``.

    Raises:
        PinsError: on digest mismatch.
    """
    actual = sha256_of(bundle_path)
    if actual != expected_sha256:
        raise PinsError(
            f"{bundle_path.name}: served bundle sha256 {actual} != pinned "
            f"{expected_sha256} — the catalog artifact was re-published; "
            "re-pin deliberately (bench-baseline discipline)"
        )


def load_pins(pins_path: Path) -> Dict[str, DirectionPins]:
    """
    Loads the pins document.

    Args:
        pins_path (Path): path to pack-acceptance/pins.json.

    Returns:
        Dict[str, DirectionPins]: direction key -> per-file sha256 map.

    Raises:
        PinsError: on a malformed document.
    """
    with open(pins_path, "r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if not isinstance(data, dict) or "leanmt" not in data or "directions" not in data:
        raise PinsError(f"{pins_path}: expected object with 'leanmt' and 'directions'")
    directions: Any = data["directions"]
    if not isinstance(directions, dict):
        raise PinsError(f"{pins_path}: 'directions' must be an object")
    for direction, entry in directions.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
            raise PinsError(f"{pins_path}: direction {direction!r} lacks a 'files' map")
    return {
        direction: DirectionPins(files=entry["files"])
        for direction, entry in directions.items()
    }


def leanmt_pin(pins_path: Path) -> str:
    """
    Reads the pinned Versta.Leanmt commit the probe binary is built from.

    Args:
        pins_path (Path): path to pack-acceptance/pins.json.

    Returns:
        str: git commit sha.
    """
    with open(pins_path, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)
    leanmt: Any = data["leanmt"]
    if not isinstance(leanmt, dict) or not isinstance(leanmt.get("commit"), str):
        raise PinsError(f"{pins_path}: 'leanmt.commit' must be a string")
    commit: Any = leanmt["commit"]
    return str(commit)


def verify_downloaded(direction_dir: Path, pins: DirectionPins) -> None:
    """
    Verifies one downloaded direction's files against its pins.

    Args:
        direction_dir (Path): the versta.download output directory.
        pins (DirectionPins): expected content sha256 per file role.

    Raises:
        PinsError: missing file, unknown file role, or hash mismatch.
    """
    with open(direction_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata: Dict[str, Any] = json.load(handle)

    errors: List[str] = []
    files: Dict[str, Any] = metadata["files"]
    for role in ("model", "vocabulary", "target_vocabulary", "shortlist"):
        name = files.get(role)
        if name is None:
            if role in pins["files"]:
                errors.append(f"missing {role} on disk but pinned")
            continue
        expected = pins["files"].get(role)
        if expected is None:
            errors.append(f"{role}={name!r} present but not pinned")
            continue
        actual = sha256_of(direction_dir / name)
        if actual != expected:
            errors.append(
                f"{role} ({name}): content sha256 {actual} != pinned {expected}"
            )
    if errors:
        raise PinsError(
            f"{direction_dir.name}: model content drifted from "
            "pack-acceptance/pins.json — re-pin deliberately in a dedicated "
            "commit (bench-baseline discipline): " + "; ".join(errors)
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify downloaded translation directions against pack-acceptance pins."
    )
    parser.add_argument(
        "--pins",
        type=Path,
        default=Path("pack-acceptance/pins.json"),
        help="Path to the pins document.",
    )
    parser.add_argument(
        "--downloads_dir",
        type=Path,
        default=Path("output"),
        help="Directory holding <src>-<tgt> download outputs.",
    )
    args = parser.parse_args()

    pins = load_pins(args.pins)
    for direction, direction_pins in sorted(pins.items()):
        verify_downloaded(args.downloads_dir / direction, direction_pins)
        print(f"pinned content verified: {direction}")
    print(f"leanmt pinned at {leanmt_pin(args.pins)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
