# speech-recognition/

whisper.cpp speech recognition models for the Versta app. Read the root
`/AGENTS.md` first; it defines the shared Python, typing and CLI conventions
this module follows.

## Module purpose

This module exports pre-quantized whisper.cpp models in native `ggml-*.bin`
format from Hugging Face and bundles them into the tarballs the Versta app
side-loads for on-device speech recognition.

- No conversion step: whisper.cpp publishes models already in the runtime format. Export here means downloading the right file, recording its metadata and packaging it.
- Every model export also downloads the Silero-VAD model (`ggml-silero-v6.2.0.bin` from `ggml-org/whisper-vad`), which the app's speech-recognition pipeline requires alongside the whisper model.
- Dependencies: `huggingface_hub` only. Models come exclusively from Hugging Face (default repo `ggerganov/whisper.cpp`; VAD from `ggml-org/whisper-vad`).

## Tools

Run from within this directory with `requirements.txt` installed.

### `versta.export`

Downloads one whisper model variant plus the VAD model and writes
`output/<model-type>/ggml-<model-type>.bin`, `ggml-silero-v6.2.0.bin` and
`metadata.json`:

```bash
python -m versta.export --model ggerganov/whisper.cpp --model-type base-q8_0 --output_dir ./output
```

- `--model`: Hugging Face repository id (default `ggerganov/whisper.cpp`).
- `--model-type`: whisper.cpp variant, used to build the filename `ggml-<model-type>.bin` (e.g. `base-q8_0` (default), `small.en`, `large-v3-turbo-q5_0`). The filename is verified against the repo file listing before downloading.
- `--languages`: supported language codes written to metadata. Defaults to the full 100-language Whisper set from `export/definitions.py`, or `["en"]` for English-only (`.en`) variants.
- Integrity: no hash verification (files carry no published hashes); sizes are recorded best-effort from the Hugging Face API.

### `versta.bundle`

Bundles a previously exported model directory into a tarball plus checksum:

```bash
python -m versta.bundle --input_dir ./output/base-q8_0 --output_dir ./output
```

- `--input_dir` (single): the directory produced by `versta.export` containing the model, VAD model and `metadata.json`. The bundle id is derived from the metadata (`whisper.<variant stem>`), so no id flag is needed.
- Output: `whisper.<variant>` tarball (`.tar.gz` + `.tar.sha256`) stamped with the module version.

## Conventions specific to this module

- Versioning here differs from the other modules: `version.txt` sits at the module root and is loaded via `versta/version.py` (exposing `VERSION`). Import `VERSION` from there rather than re-reading the file.
- The exported `metadata.json` records the model id, filenames, sizes and the supported `languages` list; `models.json` at the module root is the app-facing catalog entry (id `large-v3-turbo-q8_0`, architecture `Whisper`) pointing at `models.versta.app/speech-recognition/...`.
- Language codes are ISO-639-1 as recognized by the Whisper tokenizer (see `export/definitions.py`), including Cantonese `yue`. Keep that list byte-identical to whisper.cpp's tokenizer table.
- This module has no `language.py`: whisper language handling is a single static list in `export/definitions.py`. Keep it that way.

## Verification

No test suite. Smoke-check a change with the smallest variant and bundle it:

```bash
python -m versta.export --model-type base-q8_0
python -m versta.bundle --input_dir output/base-q8_0 --keep_intermediates --keep_input
```
