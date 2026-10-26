"""
Generate/merge the `supertonic3` entry of the voice-model feed
(`models.json`, published as `versta_text_to_speech_models.json`).

The envelope is unchanged from the Kokoro entry (id/name/base_model/size/
version/voices/architectures/bundle/checksum); `checksum` stays a URL to
the published `.sha256` sidecar. The 62 `voices` rows are emitted in the
canonical order the app asserts in `Supertonic3VoiceModelFeedTest`:
`LANGUAGES` sorted ascending, each with male then female (`VoiceGender`
declares Male first).

Removal of the legacy `versta-tts-data` (espeak/open-jtalk) entry is staged
behind ``--drop-legacy-data-entry`` and lands with workstream 5/6, not here.
"""

import argparse
import json

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from .sources import (
    BUNDLE_FILENAME,
    BUNDLE_ID,
    BUNDLE_NAME,
    FEED_BASE_URL,
    FEED_GENDER_ORDER,
    LANGUAGES,
)
from .typing import VoiceRow

LEGACY_DATA_ENTRY_ID = "versta-tts-data"


def voice_rows() -> List[VoiceRow]:
    """The 62 feed voice rows in canonical app-pinned order."""
    return [
        VoiceRow(gender=gender, language=language)
        for language in sorted(LANGUAGES)
        for gender in FEED_GENDER_ORDER
    ]


def build_feed_entry(
    version: str, bundle_bytes: int, base_url: str = FEED_BASE_URL
) -> Dict:
    """
    Builds the supertonic3 feed entry.

    Args:
        version (str): Bundle version, e.g. "v2.0.0" (also the URL path segment).
        bundle_bytes (int): Byte size of the published tarball (measured, not estimated).
        base_url (str): Publish root; the version path segment is appended.
    """
    bundle_url = f"{base_url}/{version}/{BUNDLE_FILENAME}"
    # Same envelope as the Kokoro/OCR entries: the sidecar replaces the
    # trailing ".gz" (supertonic3-bundle.tar.sha256), matching the bundler's
    # Path.with_suffix(".sha256") output name.
    checksum_url = bundle_url.removesuffix(".gz") + ".sha256"
    return {
        "id": BUNDLE_ID,
        "name": BUNDLE_NAME,
        "base_model": "Supertone/supertonic-3",
        "size": bundle_bytes,
        "version": version,
        "voices": voice_rows(),
        "architectures": ["Supertonic3"],
        "bundle": bundle_url,
        "checksum": checksum_url,
    }


def merge_feed(
    feed_path: Path,
    version: str,
    bundle_bytes: int,
    drop_legacy_data_entry: bool = False,
) -> Path:
    """
    Idempotently merges the supertonic3 entry into the checked-in feed.

    Args:
        feed_path (Path): Path of models.json.
        version (str): Bundle version.
        bundle_bytes (int): Measured tarball size.
        drop_legacy_data_entry (bool): Also remove the legacy
            `versta-tts-data` entry — staged for the 5/6 app release, off
            by default here.

    Raises:
        ValueError: If the feed is malformed or a conflicting `supertonic3`
            entry with different content would be silently overwritten.
    """
    with open(feed_path, encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError(f"{feed_path} does not contain a JSON array")

    entry = build_feed_entry(version, bundle_bytes)
    replaced = False
    for index, existing in enumerate(entries):
        if existing.get("id") == BUNDLE_ID:
            if existing != entry:
                entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    if drop_legacy_data_entry:
        entries = [e for e in entries if e.get("id") != LEGACY_DATA_ENTRY_ID]

    with open(feed_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    print(f"feed: {feed_path} updated ({len(entries)} entries)")
    return feed_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        prog="versta.supertonic3 feed",
        description="Merge the supertonic3 entry into the voice-model feed.",
    )
    parser.add_argument(
        "--feed",
        type=Path,
        default=Path("models.json"),
        help="Path of the checked-in voice-model feed JSON.",
    )
    parser.add_argument("--version", required=True, help="Bundle version, e.g. v2.0.0.")
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="The published tarball (size is measured from the file).",
    )
    parser.add_argument(
        "--drop_legacy_data_entry",
        action="store_true",
        help="Also remove versta-tts-data (staged for the 5/6 app release).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_feed(
        args.feed, args.version, args.bundle.stat().st_size, args.drop_legacy_data_entry
    )


if __name__ == "__main__":
    main()
