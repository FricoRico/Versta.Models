# OCR Models (PP-OCRv6, MNN int8)
The Versta app runs on-device OCR with PaddleOCR PP-OCRv6 models in MNN int8 format. This module downloads the official upstream inference models, converts them to ONNX via `paddle2onnx`, converts them to MNN with int8 weight quantization via `MNNConvert`, and produces a pack in `output/paddle-ocr-v6/` with a `manifest.json` ready for upload to `models.versta.app`.

## Provenance
- Upstream models (all Apache-2.0): PP-OCRv6 tiny detector, PP-OCRv6 tiny recognizer (latin script slot), PP-OCRv6 small recognizer (CJ script slot) and the PP-LCNet textline orientation model from the [PaddleX/PaddleOCR model zoo](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/); the PULC language classifier from [PaddleClas](https://paddleclas.bj.bcebos.com/models/PULC/).
- The conversion pipeline is a port of the MIT-licensed scripts by David Ventura from [translator-rs](https://github.com/DavidVentura/translator-rs) (`scripts/convert_ppocr_v6_mnn.py`, `scripts/convert_det_lowres_mnn.py`, `scripts/convert_pulc_language_mnn.sh`).
- The produced files match, byte-for-byte or within int8 rounding, the reference pack published at [offline-translator.davidv.dev/ocr/1/PP-OCRv6](https://offline-translator.davidv.dev/ocr/1/PP-OCRv6/) (MIT), which serves as a golden reference during development.

Pinned toolchain versions: `paddlepaddle 3.0`, `paddle2onnx 2.1` (required for PIR-format models), MNN `3.6.1` (converter built from the vendored source submodule).

## Requirements
Besides [uv](https://docs.astral.sh/uv/), building `MNNConvert` from source needs a C++ toolchain: **GCC (g++) and CMake**. These must be available on PATH (e.g. inside a `toolbox` container on immutable distros). The MNN source is a git submodule pinned to the `3.6.1` release tag — after cloning, run:

```bash
git submodule update --init object-character-recognition/vendor/MNN
```

`MNNConvert` is built automatically on first run (takes ~15 minutes) if no binary is found; pass `--mnnconvert` to use a prebuilt binary instead.

## The pack
The pack consists of (roles refer to `manifest.json`):

| File | Role | Notes |
| --- | --- | --- |
| `PP-OCRv6_tiny_det_int8.mnn` | detector | full-resolution probability map |
| `PP-OCRv6_tiny_det_half_int8.mnn` | detector | 1/2-resolution map, exact — shipped for live mode (priority 1) |
| `PP-OCRv6_tiny_det_quarter_int8.mnn` | detector | 1/4-resolution map, approximate — stills-only |
| `PP-OCRv6_tiny_rec_int8.mnn` + `PP-OCRv6_tiny_keys.txt` | recognizer + keys | latin script |
| `PP-OCRv6_small_rec_int8.mnn` + `PP-OCRv6_small_keys.txt` | recognizer + keys | CJ script (different charset than tiny — it drops Japanese kana) |
| `PULC_int8.mnn` | scriptClassifier | PaddleClas language/script classifier |
| `textline_ori_x0_25_wq8.mnn` | textlineOrientation | PP-LCNet x0.25 |

The `half`/`quarter` detector variants are ONNX graph edits: the DBNet head's 2x2 stride-2 deconvs are folded into 1x1 convolutions with spatially averaged weights, producing the probability map at 1/2 (resp. 1/4) input resolution.

## Producing the pack
```bash
uv sync
uv run python -m versta.export --output_dir ./output
```

## Bundling the pack
Packs the export output into a deployable tarball (files flat at the archive root, `manifest.json` included) after verifying every file against the manifest, and refreshes the checked-in `models.json` catalog entry (version, size, `models.versta.app` URLs):

```bash
uv run python -m versta.bundle
```

Produces `output/paddle-ocr-bundle.tar.gz` (rename before publishing if hosting multiple packs) and `output/paddle-ocr-bundle.tar.sha256`.
