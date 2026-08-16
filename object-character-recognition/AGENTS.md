# object-character-recognition/

PaddleOCR detection/recognition models for the Versta app. Read the root
`/AGENTS.md` first; it defines the shared Python, typing and CLI conventions
this module follows.

## Module purpose

This module exports PaddleOCR models to the on-device runtime format (ONNX →
ORT) and bundles them for side-loading into the Versta app. It also contains
the evaluation harness used to compare OCR engines against OCR benchmarks.

- Supported upstream: PaddleOCR(e.g. `PaddlePaddle/PP-OCRv5_mobile_rec`). The export path may generalize to other models, but only PaddleOCR is supported today — guard against undocumented formats.
- Two module kinds exist: `detector` and `recognizer`. Recognizers carry a vocabulary/tokenizer file; detectors must not. Enforce that coupling in code (see `export/__main__.py`Step 4) rather than discovering mismatches at bundle time.
- Heavy dependencies: `paddlepaddle`, `paddlex`, `paddleocr`, `paddle2onnx`, `paddleslim`, `onnxsim`, `onnxruntime`; the evaluation tool adds `datasets`, `pandas`, `python-levenshtein`, `rapidfuzz`, `langchain`, `pdfplumber`, `python-dotenv`. This module has its own `.venv` (Python 3.11) — do not share environments across modules.

## Tools

Run from within this directory after `uv sync` (dependencies are declared in `pyproject.toml`).

### `versta.export`

Converts a downloaded PaddleOCR checkpoint to ORT:

```bash
uv run python -m versta.export --model PaddlePaddle/PP-OCRv5_mobile_rec --module recognizer --export_dir ./export
```

- `--model` (required): Hugging Face repository id. The repo is validated via an HTTP HEAD before `snapshot_download`.
- `--module`: `detector` or `recognizer`. Note the argparse default is currently `recognizer` while the help text and `main()` signature say `detector` — a known inconsistency; fix deliberately if you touch this flag rather than assuming either side is canonical. Output lands in `<export_dir>/<model-name-lowercased>/`.
- `--export_dir`: defaults to `export/` (gitignored). Pipeline: ONNX (FP16) → [onnxsim simplify — currently disabled/commented] → ORT + tokenizer/vocab save → metadata.
- `--keep_intermediates`, `--clear_cache`: same semantics as the other export tools.
- When changing the pipeline (e.g. re-enabling simplification via `quantize.py:simplify_model`), keep intermediates under `intermediates/` and final artifacts (`.ort`, vocab, `metadata.json`) at the model output dir.

### `versta.bundle`

Bundles converted detector/recognizer directories into one tarball + `.sha256`:

```bash
uv run python -m versta.bundle --input_dir ./export/pp-ocrv5_mobile_rec --unique_id paddle-ocr
```

- Accepts multiple, mixed module types in one bundle (`language.py:extract_unique_modules`).
- The checked-in `models.json` (id `paddle-ocr`, architecture `PaddleOCR`) lists the supported language set of the bundled recognizer vocabulary; regenerate it when the bundled models change.

### `versta.evaluate`

OmniDocBench-aligned evaluation comparing PaddleOCR against the exported ONNX/
ORT pipeline:

```bash
# Reference PaddleOCR run
uv run python -m versta.evaluate --output-dir ./output --model-type paddleocr

# Exported ONNX/ORT engines
uv run python -m versta.evaluate --output-dir ./output --model-type onnx \
  --detector path/detector.ort --recognizer path/recognizer.ort

# Score previously generated predictions only
uv run python -m versta.evaluate --output-dir ./output --eval-only
```

- Note: this CLI uses dashed flags (`--output-dir`, `--model-type`, `--eval-only`) unlike the snake_case flags elsewhere in the repo — follow the local precedent of this file when editing it; do not restyle wholesale.
- Environment: a `.env` file next to `evaluate/__main__.py` is loaded via `python-dotenv`; `FLAGS_use_mkldnn=0` and `OMNIDOCBENCH_PDFLATEX=pdftex` are set in-process. Keep benchmark configuration in `.env`/env vars, not hard-coded.
- Predictions land in `results/predictions/...` (gitignored); scoring reuses the checked-in `results` tree only for reference metrics, never commit new predictions.
- Structure: `onnx_engine.py` + `pipeline.py` implement inference, `inference.py`/`run_eval.py` drive runs, `dataset.py` loads the benchmark, `compare.py`/`generate_score.py` score outputs. Engine-neutral types live in `evaluate/typing.py` — keep them fully typed.

## Conventions specific to this module

- `version.txt` lives at `versta/version.txt` (currently `v1.0.0`), read at import time by each `__main__.py`.
- Paddle stack environment quirks (e.g. disabling MKLDNN) belong at process entrypoints (`os.environ` before importing paddle), not scattered through library code.
- Strict typing holds here as everywhere: `export/typing.py` and `evaluate/typing.py` define the boundary-crossing TypedDicts; extend them when adding fields instead of passing loosely typed dicts.

## Verification

No test suite; a stale pytest cache references an `onnx_engine` test that is
not part of the tree — ignore it. Smoke-check export with a small recognizer
export and, when touching evaluation, an `--eval-only` scoring run over
existing predictions.
