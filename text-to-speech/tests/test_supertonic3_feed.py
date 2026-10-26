"""Feed entry generation: 62 rows, canonical order, unchanged envelope."""

import json

import pytest

from versta.supertonic3.feed import (
    LEGACY_DATA_ENTRY_ID,
    build_feed_entry,
    merge_feed,
    voice_rows,
)
from versta.supertonic3.sources import LANGUAGES


def test_voice_rows_are_62_in_canonical_order():
    rows = voice_rows()
    assert len(rows) == 62
    expected = [
        {"gender": gender, "language": language}
        for language in sorted(LANGUAGES)
        for gender in ("male", "female")
    ]
    assert [dict(row) for row in rows] == expected


def test_feed_entry_envelope_matches_kokoro_shape():
    entry = build_feed_entry("v2.0.0", 119000000)
    assert set(entry.keys()) == {
        "id",
        "name",
        "base_model",
        "size",
        "version",
        "voices",
        "architectures",
        "bundle",
        "checksum",
    }
    assert entry["id"] == "supertonic3"
    assert entry["architectures"] == ["Supertonic3"]
    assert entry["size"] == 119000000
    assert entry["bundle"] == (
        "https://models.versta.app/text-to-speech/v2.0.0/supertonic3-bundle.tar.gz"
    )
    assert entry["checksum"] == (
        "https://models.versta.app/text-to-speech/v2.0.0/supertonic3-bundle.tar.sha256"
    )
    assert len(entry["voices"]) == 62


def _seed_feed(tmp_path):
    feed = tmp_path / "models.json"
    feed.write_text(
        json.dumps(
            [
                {"id": "kokoro", "version": "v1.2.0", "voices": []},
                {"id": LEGACY_DATA_ENTRY_ID, "version": "v1.0.0"},
            ]
        )
    )
    return feed


def test_merge_appends_and_is_idempotent(tmp_path):
    feed = _seed_feed(tmp_path)
    merge_feed(feed, "v2.0.0", 100)
    first = json.loads(feed.read_text())
    merge_feed(feed, "v2.0.0", 100)
    second = json.loads(feed.read_text())
    assert first == second
    assert [e["id"] for e in second] == ["kokoro", LEGACY_DATA_ENTRY_ID, "supertonic3"]


def test_merge_replaces_conflicting_entry(tmp_path):
    feed = _seed_feed(tmp_path)
    merge_feed(feed, "v2.0.0", 100)
    merge_feed(feed, "v2.0.0", 200)
    entries = json.loads(feed.read_text())
    assert entries[-1]["size"] == 200
    assert [e["id"] for e in entries].count("supertonic3") == 1


def test_legacy_data_entry_only_drops_when_flagged(tmp_path):
    feed = _seed_feed(tmp_path)
    merge_feed(feed, "v2.0.0", 100, drop_legacy_data_entry=False)
    assert LEGACY_DATA_ENTRY_ID in [e["id"] for e in json.loads(feed.read_text())]
    merge_feed(feed, "v2.0.0", 100, drop_legacy_data_entry=True)
    assert LEGACY_DATA_ENTRY_ID not in [e["id"] for e in json.loads(feed.read_text())]


def test_merge_rejects_malformed_feed(tmp_path):
    feed = tmp_path / "models.json"
    feed.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        merge_feed(feed, "v2.0.0", 100)
