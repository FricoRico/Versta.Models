# text-to-speech/

Kokoro and Piper text-to-speech models for the Versta app. Read the root
`/AGENTS.md` first; it defines the shared Python, typing and CLI conventions
this module follows.

## Module purpose

This module converts TTS models from Hugging Face into the runtime formats the
Versta app needs (ONNX → quantized ONNX → ORT for on-device inference via
onnxruntime), and bundles them for side-loading.

- Two model families are supported: **Kokoro** (e.g. `hexgrad/Kokoro-82M`) and **Piper** voices (`rhasspy/piper-voices`). Format-specific code lives in `export/convert_kokoro_to_onnx.py` / `convert_piper_to_onnx.py` and `quantize_kokoro.py` / `quantize_piper.py`, dispatched through the format-agnostic `convert_onnx.py` / `quantize.py`.
- The export ships more than weights: tokenizer files and voice metadata are saved into the output so the bundle is fully self-contained for offline TTS.
- Heavy dependencies: `onnxruntime` + `onnxruntime-tools`, `optimum[exporters]`, `transformers`, `kokoro==0.8.4` (exactly pinned), `huggingface_hub`, `pycountry`. Pin style stays `>=,<` per the root conventions; Kokoro is the deliberate exception.

## Tools

Run from within this directory with `requirements.txt` installed.

### `versta.export`

Converts one model end-to-end: ONNX export → quantization → ORT conversion →
tokenizer/voice extraction → metadata generation:

```bash
python -m versta.export --model hexgrad/Kokoro-82M --model_format kokoro --output_dir ./output
python -m versta.export --model rhasspy/piper-voices --model_format piper --voice nl/nl_NL/mls/medium
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
python -m versta.bundle --input_dir ./output/kokoro --unique_id kokoro --output_dir ./output
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

No test suite. Smoke-check a change with a Piper voice (smallest download) or
Kokoro, and confirm the ORT files, tokenizer files and `metadata.json`:

```bash
python -m versta.export --model rhasspy/piper-voices --model_format piper --voice nl/nl_NL/mls/medium --keep_intermediates
python -m versta.bundle --input_dir <exported dir> --unique_id piper-nl-nl_NL-mls-medium
```
