# Versta.Models
This repository contains tooling to easily download, bundle and deploy AI models to be compatible with the Versta mobile app.

Each top-level directory is a self-contained [uv](https://docs.astral.sh/uv/) project producing the model bundles the app side-loads from `models.versta.app`:

- [translation/](translation) — Firefox (Bergamot) translation models, downloaded and bundled per language pair.
- [object-character-recognition/](object-character-recognition) — PaddleOCR PP-OCRv6 models converted to MNN int8, bundled as OCR packs (detector, recognizers, script classifier, textline orientation).
- [speech-recognition/](speech-recognition) — whisper.cpp (ggml) speech recognition models.
- [text-to-speech/](text-to-speech) — Kokoro/Piper TTS models converted to ONNX/ORT.
- [data/](data) — shared runtime data bundles (espeak-ng, open-jtalk).

Module-specific usage lives in each module's README.md.
