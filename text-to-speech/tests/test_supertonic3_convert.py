"""Conversion output canonicalization: the mnn_uuid determinism contract."""

import pytest

from versta.supertonic3.convert_to_mnn import NIL_UUID, canonicalize_mnn_uuid


def test_canonicalize_replaces_random_uuid(tmp_path):
    model = tmp_path / "stage.mnn"
    head = b"\x00" * 56 + b"$t\x00"
    uuid = b"eec8803f-7555-43c1-8bbc-818d8a0ce062"
    tail = b"\xaa\xbb" * 128
    model.write_bytes(head + uuid + tail)

    canonicalize_mnn_uuid(model)
    assert model.read_bytes() == head + NIL_UUID + tail


def test_canonicalize_is_idempotent(tmp_path):
    model = tmp_path / "stage.mnn"
    payload = b"\x00" * 16 + NIL_UUID + b"\xff" * 64
    model.write_bytes(payload)

    canonicalize_mnn_uuid(model)
    assert model.read_bytes() == payload


def test_canonicalize_fails_closed_without_a_uuid(tmp_path):
    model = tmp_path / "stage.mnn"
    model.write_bytes(b"no uuid here")
    with pytest.raises(RuntimeError, match="mnn_uuid"):
        canonicalize_mnn_uuid(model)


def test_canonicalize_fails_closed_on_duplicate_uuids(tmp_path):
    model = tmp_path / "stage.mnn"
    uuid = b"eec8803f-7555-43c1-8bbc-818d8a0ce062"
    model.write_bytes(uuid + b"\x00" * 8 + uuid)
    with pytest.raises(RuntimeError, match="mnn_uuid"):
        canonicalize_mnn_uuid(model)
