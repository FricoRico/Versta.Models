"""Upstream source pins: the download pattern set covers every pinned file."""

from versta.supertonic3.convert_to_mnn import ALL_UPSTREAM_PATTERNS
from versta.supertonic3.sources import (
    EXPECTED_SHA256,
    LICENSE_UPSTREAM_PATH,
    ONNX_FILES,
    PASSTHROUGH_FILES,
    VOICE_STYLE_FILES,
)


def test_download_patterns_are_upstream_paths():
    upstream_paths = {
        *ONNX_FILES.values(),
        *PASSTHROUGH_FILES,
        *VOICE_STYLE_FILES,
        LICENSE_UPSTREAM_PATH,
    }
    assert set(ALL_UPSTREAM_PATTERNS) == upstream_paths
    assert all(
        path in EXPECTED_SHA256 or path in ONNX_FILES.values()
        for path in upstream_paths
    )
