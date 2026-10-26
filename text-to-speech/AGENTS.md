# text-to-speech/

Supertonic 3 and Piper text-to-speech models for the Versta app. Read the root
`/AGENTS.md` first; it defines the shared Python, typing and CLI conventions
this module follows.

## Module purpose

This module ships two paths:

- **Supertonic 3 → MNN int8** (`versta/supertonic3/`, bundle v2.0.0): the current
  shipped voice bundle — 4-stage MNN conversion, seeded numerics verification,
  OpenRAIL-compliant bundling, feed entry, conversion ledger. Usage and gates:
  `README.md`.
- **Piper → ONNX/ORT legacy** (`versta/export|bundle|batch/`), unchanged. Kokoro
  is archived in `versta/export/_archive/` (NEU-68; `convert_onnx.py` /
  `quantize.py` fail closed with a pointer for `--model_format kokoro`).

- Heavy dependencies: `onnxruntime` + `onnxruntime-tools`, `optimum[exporters]`, `transformers`, `huggingface_hub`, `pycountry`, `numpy`. Pin style stays `>=,<` per the root conventions.
- `vendor/mnn` pins MNN at tag 3.6.1 (the NEU-64-validated version) and provides the MNNConvert build the Supertonic 3 converter drives.

## Tools

Run from within this directory after `uv sync` (dependencies are declared in `pyproject.toml`).

### `versta.export`

Converts one model end-to-end: ONNX export → quantization → ORT conversion →
tokenizer/voice extraction → metadata generation:

```bash
uv run python -m versta.export --model hexgrad/Kokoro-82M --model_format kokoro --output_dir ./output
uv run python -m versta.export --model rhasspy/piper-voices --model_format piper --voice nl/nl_NL/mls/medium
```

- `--model` (required): Hugging Face model name (Kokoro) or `rhasspy/piper-voices` (Piper).
- `--model_format`: `kokoro` (default) or `piper`. Anything else must raise — do not add silent fallbacks.
- `--voice` (Piper only): voice path inside the Piper repo (e.g. `nl/nl_NL/mls/medium`).
- `--output_dir`: defaults to `output/kokoro`; the final layout is resolved by `export/utils.py:output_folder` per model/format/voice.
- `--keep_intermediates` (`store_true`): keep the `intermediates/converted` and `intermediates/quantized` staging dirs.
- `--clear_cache` (`store_true`): remove the model's Hugging Face cache directory after export.

Pipeline order in `main()` is contractually significant (convert → quantize → ORT → tokenizer → voices → metadata → cleanup); the batch tool depends on it.

### `versta.bundle`

Bundles one or more converted model directories into a tarball plus `.sha256`
checksum:

```bash
uv run python -m versta.bundle --input_dir ./output/kokoro --unique_id kokoro --output_dir ./output
```

- `--unique_id`: model id written into the bundle metadata.
- `--input_dir`: a single directory produced by `versta.export` (unlike the translation bundler, this flag does not take `nargs="+"`).
- Language-pair validation (`language.py:validate_translation_pairs`) is shared with the other bundlers; use it exactly as they do.

### `versta.batch`

Batch export + bundle driven by a JSON input file
(`batch/export.py`, `batch/model_file.py`, `batch/typing.py`), producing a
deployable `models.json` catalog in the same shape as the checked-in one
(e.g. id `kokoro`, `voices` list with `gender`/`language`).

## Conventions specific to this module

- `version.txt` lives at `versta/version.txt` and is read at import time in each `__main__.py` (currently `v1.2.0`). Bump it when the bundle format or metadata schema changes.
- Quantization is format-specific: route new formats through `quantize.py` (`quantize_model(converted_dir, "model.onnx", quantization_dir, model_format)`), adding a format key rather than branching inside the format implementations.
- Voice handling must keep language→gender mapping data-driven (via `pycountry`, `export/metadata.py:get_voices`); never hard-code voice/language pairs in multiple places.
- ORT is the final on-device format. ONNX files are intermediates: they stay under `intermediates/` unless `--keep_intermediates` is passed.
- Strict typing holds here as everywhere: extend `export/typing.py`/`batch/typing.py` TypedDicts when adding fields; no loosely typed dicts crossing function boundaries.

## Verification

- `tests/` carries a pytest suite (`uv run pytest tests/`) covering the
  Supertonic 3 metadata contract, feed row order, bundle lint/tarball and
  ledger behavior. Extend it with any schema touch.
- The Supertonic 3 numerics gate (`versta.supertonic3 verify`) is the
  behavioral check for conversion changes; it runs seeded and fails closed.
- Piper smoke-check stays CLI-level (smallest download), confirming the ORT
  files, tokenizer files and `metadata.json`:

```bash
uv run python -m versta.export --model rhasspy/piper-voices --model_format piper --voice nl/nl_NL/mls/medium --keep_intermediates
uv run python -m versta.bundle --input_dir <exported dir> --unique_id piper-nl-nl_NL-mls-medium
```
