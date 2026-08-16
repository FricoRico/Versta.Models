# object-character-recognition/

PaddleOCR detection/recognition models for the Versta app. Read the root
`/AGENTS.md` first; it defines the shared Python, typing and CLI conventions
this module follows.

## Module purpose

This module exports PaddleOCR models to the on-device runtime format (MNN,
int8 weight-quantized) and bundles them for side-loading into the Versta app.

- Supported upstream: the official PP-OCRv6/PaddleX inference tars (PIR format) plus the PULC language classifier (PaddleClas, legacy pdmodel format). The catalog lives in `export/definitions.py` — guard against undocumented formats.
- Four module kinds exist: `detector`, `recognizer`, `scriptClassifier`, `textlineOrientation`. Recognizers carry a per-tier character dictionary (`*_keys.txt`); the tiny and small tiers have different charsets (tiny drops Japanese kana) — always extract each tier's own dictionary from its `inference.yml`.
- Heavy dependencies: `paddlepaddle`, `paddle2onnx` (>=2.1, required for PIR), `onnx`, `pyyaml`, `tqdm`. This module has its own `.venv` (Python 3.11) — do not share environments across modules.
- Native toolchain: `MNNConvert` is built from the vendored MNN source submodule (`vendor/MNN`, pinned to the `3.6.1` release tag) and needs **GCC (g++) and CMake** on PATH. On immutable distros run it inside a toolbox container.

## Tools

Run from within this directory after `uv sync` (dependencies are declared in `pyproject.toml`).

### `versta.export`

Downloads the official upstream tars and converts them to MNN int8, producing the PP-OCRv6 pack:

```bash
uv run python -m versta.export --output_dir ./output
```

- Output lands in `<output_dir>/paddle-ocr-v6/`: three detector variants (full, `half` — exact, shipped for live mode — and `quarter`, stills-only), both recognizer tiers with their keys files, `PULC_int8.mnn`, `textline_ori_x0_25_wq8.mnn`, and `manifest.json` (per-file name/sizeBytes/sha256/role/script/priority).
- Pipeline per model: download tar → extract → `paddle2onnx` (opset 14 for PIR, opset 11 for PULC) → `MNNConvert -f ONNX --bizCode biz --weightQuantBits 8`. Detector variants fold the DBNet head deconvs into 1x1 convolutions (`export/fold_deconv.py`, ported from translator-rs, MIT).
- `--models` restricts the run to specific tar stems; `--mnnconvert` points at a prebuilt converter binary (otherwise the vendored submodule is built on first use); `--keep_intermediates` keeps tars/extracted models/ONNX.

### `versta.bundle`

Bundles converted detector/recognizer directories into one tarball + `.sha256`:

```bash
uv run python -m versta.bundle --input_dir ./export/pp-ocrv5_mobile_rec --unique_id paddle-ocr
```

- Accepts multiple, mixed module types in one bundle (`language.py:extract_unique_modules`).
- The checked-in `models.json` (id `paddle-ocr`, architecture `PaddleOCR`) describes the legacy PP-OCRv5 ORT pack and is superseded by the MNN pack's `manifest.json`; it stays until the app consumes MNN packs.

## Conventions specific to this module

- `version.txt` lives at `versta/version.txt` (currently `v1.1.0`), read at import time by each `__main__.py`.
- Paddle stack environment quirks (e.g. disabling MKLDNN) belong at process entrypoints (`os.environ` before importing paddle), not scattered through library code.
- Strict typing holds here as everywhere: `export/typing.py` defines the boundary-crossing TypedDicts; extend them when adding fields instead of passing loosely typed dicts.

## Verification

No test suite. Smoke-check export with a run restricted to a single model,
e.g. `uv run python -m versta.export --models PP-OCRv6_tiny_det_infer`.
