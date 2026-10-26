"""metadata.json writer: the exact app-load contract (NEU-70 shape)."""

import json

from versta.supertonic3.metadata import build_metadata, write_metadata


def test_metadata_matches_app_contract(tmp_path):
    metadata = build_metadata("v2.0.0")

    assert metadata["version"] == "v2.0.0"
    assert metadata["base_model"] == "Supertone/supertonic-3"
    assert metadata["architectures"] == ["Supertonic3"]

    files = metadata["files"]
    assert files["inference"]["models"] == {
        "duration_predictor": "duration_predictor.mnn",
        "text_encoder": "text_encoder.mnn",
        "vector_estimator": "vector_estimator.mnn",
        "vocoder": "vocoder.mnn",
    }
    assert files["inference"]["config"] == "tts.json"
    assert files["tokenizer"] == {"unicode_indexer": "unicode_indexer.json"}
    assert files["voices"] == [
        f"voice_styles/{name}.json"
        for name in ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
    ]


def test_written_metadata_round_trips(tmp_path):
    path = write_metadata(tmp_path, "v2.0.0")
    with open(path, encoding="utf-8") as handle:
        reloaded = json.load(handle)
    assert reloaded == build_metadata("v2.0.0")
    # Paths stay relative to the model root — no leading slash segments.
    for file_path in reloaded["files"]["inference"]["models"].values():
        assert not file_path.startswith("/")
