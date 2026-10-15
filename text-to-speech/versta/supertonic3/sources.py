"""
Upstream source pins for `Supertone/supertonic-3`.

Fail-fast rule: the pipeline never consumes a moving ref. The revision is a
required CLI argument; every passthrough file is verified against the sha256
below after download. The hashes were captured from the files validated
byte-identical to upstream in the NEU-64 spike and are the contract for
"ship verbatim, pinned" (2/6 design delta 3).
"""

from typing import Dict, List


UPSTREAM_REPO = "Supertone/supertonic-3"

#: Upstream files converted to MNN, per stage (paths inside the repo).
ONNX_FILES: Dict[str, str] = {
    "duration_predictor": "onnx/duration_predictor.onnx",
    "text_encoder": "onnx/text_encoder.onnx",
    "vector_estimator": "onnx/vector_estimator.onnx",
    "vocoder": "onnx/vocoder.onnx",
}

#: Files shipped byte-identical in the bundle: upstream path → bundle path.
PASSTHROUGH_FILES: Dict[str, str] = {
    "onnx/tts.json": "tts.json",
    "onnx/unicode_indexer.json": "unicode_indexer.json",
}

#: The ten voice-style files, upstream path → bundle path.
VOICE_STYLE_FILES: Dict[str, str] = {
    f"voice_styles/{name}.json": f"voice_styles/{name}.json"
    for name in ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
}

LICENSE_UPSTREAM_PATH = "LICENSE"
LICENSE_BUNDLE_PATH = "LICENSE"

#: sha256 of every passthrough/style/license file at the NEU-64-validated
#: upstream revision. A mismatch means upstream moved content under us:
#: stop, do not silently ship different bytes.
EXPECTED_SHA256: Dict[str, str] = {
    "onnx/tts.json": "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    "onnx/unicode_indexer.json": "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
    "LICENSE": "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
    "voice_styles/F1.json": "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2",
    "voice_styles/F2.json": "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6",
    "voice_styles/F3.json": "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab",
    "voice_styles/F4.json": "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b",
    "voice_styles/F5.json": "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d",
    "voice_styles/M1.json": "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b",
    "voice_styles/M2.json": "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50",
    "voice_styles/M3.json": "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b",
    "voice_styles/M4.json": "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303",
    "voice_styles/M5.json": "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2",
}

#: Languages with Supertonic 3 coverage. This is the single source for the
#: 62 feed rows (31 languages × 2 genders) and must match the app-side
#: `Supertonic3.LANGUAGES` constant asserted by `Supertonic3VoiceModelFeedTest`
#: (languages without coverage: ca, fa, he, zh).
LANGUAGES: List[str] = [
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
]

#: Gender order per language row in the feed, mirroring the app enum order
#: (`VoiceGender.entries` declares Male before Female; the feed test builds
#: rows in exactly this order).
FEED_GENDER_ORDER: List[str] = ["male", "female"]

BUNDLE_ID = "supertonic3"
BUNDLE_NAME = "Supertonic 3"
BUNDLE_FILENAME = "supertonic3-bundle.tar.gz"
FEED_BASE_URL = "https://models.versta.app/text-to-speech"
