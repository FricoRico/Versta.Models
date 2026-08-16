# object-character-recognition/

PaddleOCR detection/recognition models for the Versta app. Read the root
`/AGENTS.md` first; it defines the shared Python, typing and CLI conventions
this module follows.

## Module purpose

This module exports PaddleOCR models to the on-device runtime format (MNN,
int8 weight-quantized) and bundles them for side-loading into the Versta app.

- Supported upstream: the official PP-OCRv6/PaddleX inference tars (PIR format), the PULC language classifier (PaddleClas, legacy pdmodel format), and the DocAligner `lcnet050` corner-regression model (DocsaidLab; ready ONNX from an HTTP mirror, no tar/paddle2onnx step). The catalog lives in `export/definitions.py` — guard against undocumented formats.
- Five module kinds exist: `detector`, `recognizer`, `scriptClassifier`, `textlineOrientation`, `aligner`. Recognizers carry a per-tier character dictionary (`*_keys.txt`); the tiny and small tiers have different charsets (tiny drops Japanese kana) — always extract each tier's own dictionary from its `inference.yml`. Aligners skip the Paddle toolchain entirely (`pipeline.py` branches on kind).
- Heavy dependencies: `paddlepaddle`, `paddle2onnx` (>=2.1, required for PIR), `onnx`, `pyyaml`, `tqdm`. This module has its own `.venv` (Python 3.11) — do not share environments across modules.
- Native toolchain: `MNNConvert` is built from the vendored MNN source submodule (`vendor/mnn`, pinned to the `3.6.1` release tag) and needs **GCC (g++) and CMake** on PATH. On immutable distros run it inside a toolbox container.

## Tools

Run from within this directory after `uv sync` (dependencies are declared in `pyproject.toml`).

### `versta.export`

Downloads the official upstream tars and converts them to MNN int8, producing the PP-OCRv6 pack:

```bash
uv run python -m versta.export --output_dir ./output
```

- Output lands in `<output_dir>/paddle-ocr-v6/`: three detector variants (full, `half` — exact, shipped for live mode — and `quarter`, stills-only), both recognizer tiers with their keys files, `PULC_int8.mnn`, `textline_ori_x0_25_wq8.mnn`, `docaligner_lcnet050_int8.mnn`, and `manifest.json` (per-file name/sizeBytes/sha256/role/script/priority). The manifest merges by file name, so `--models` re-runs update entries without wiping the rest of the pack.
- Pipeline per model: download tar → extract → `paddle2onnx` (opset 14 for PIR, opset 11 for PULC) → `MNNConvert -f ONNX --bizCode biz --weightQuantBits 8`. Detector variants fold the DBNet head deconvs into 1x1 convolutions (`export/fold_deconv.py`, ported from translator-rs, MIT).
- `--models` restricts the run to specific tar stems; `--mnnconvert` points at a prebuilt converter binary (otherwise the vendored submodule is built on first use); `--keep_intermediates` keeps tars/extracted models/ONNX.

### `versta.train.glyphmatte`

Trains the glyph-matte U-Net and exports `glyphmatte_int8.mnn` into the pack
(manifest role `glyphmatte`). Sub-CLIs: `assets` (pinned fonts/word lists into
`assets/`), `gen_data` (inspection sheets), `train` (GPU; needs the `rocm` or
`cu130` extra — `uv sync --extra rocm`; ROCm training additionally needs a C++
toolchain at runtime for MIOpen/HIPRTC: run inside a toolbox container on
immutable distros), `export_onnx` (four named outputs: matte, weight,
foreground, background), `convert_mnn` (spawns the bundle's vendored
MNNConvert and merges into the pack manifest), `eval` (synthetic-val metrics
+ ONNX↔int8 drift; auto-applies the PyMNN execstack patch on first use).

### `versta.bundle`

Bundles a converted OCR pack into one tarball + `.sha256` and refreshes the
checked-in `models.json` catalog entry:

```bash
uv run python -m versta.bundle
```

- `--input_dir` defaults to `output/<PACK_NAME>` (the export pack); every manifest entry is verified (exists, size, sha256) before bundling. Files are stored flat at the archive root, including `manifest.json`.
- Produces `<output_dir>/<unique_id>-bundle.tar.gz` + `.tar.sha256` and updates `models.json` in place: version (from `versta/version.txt`), tarball size, and `models.versta.app` bundle/checksum URLs. `id`, `name`, `architectures` and `languages` are preserved.
- Stdlib only; the legacy multi-input-dir metadata extraction (`metadata.py`/`language.py`) was removed with the ORT export.

## Conventions specific to this module

- `version.txt` lives at `versta/version.txt` (currently `v1.1.0`), read at import time by each `__main__.py`.
- Paddle stack environment quirks (e.g. disabling MKLDNN) belong at process entrypoints (`os.environ` before importing paddle), not scattered through library code.
- Strict typing holds here as everywhere: `export/typing.py` defines the boundary-crossing TypedDicts; extend them when adding fields instead of passing loosely typed dicts.

## Verification

No test suite. Smoke-check export with a run restricted to a single model,
e.g. `uv run python -m versta.export --models PP-OCRv6_tiny_det_infer`.

`versta.export.check_docaligner` validates the DocAligner int8 MNN export
against CPU onnxruntime on synthetic document photos (corner points must
agree within a few px on the 256 grid). Its dependencies live in the `dev`
group (`onnxruntime`, `pillow`, `MNN`): `uv sync --group dev`. The PyMNN
wheel ships an execstack-flagged `_mnncengine*.so` — on kernels that reject
it, clear the X bit of the ELF `PT_GNU_STACK` header in-place.
