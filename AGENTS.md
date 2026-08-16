# Versta.Models

This file describes the project's conventions. Read it before writing or
modifying code. More specific conventions live in the nested AGENTS.md files:

- `/translation/AGENTS.md` — Firefox/Bergamot translation model download, bundling and batch export.
- `/speech-recognition/AGENTS.md` — whisper.cpp (ggml) speech recognition model export and bundling.
- `/text-to-speech/AGENTS.md` — Kokoro/Piper TTS model conversion (ONNX → ORT), quantization and bundling.
- `/object-character-recognition/AGENTS.md` — PaddleOCR model export and OmniDocBench-aligned evaluation.
- `/data/AGENTS.md` — shared runtime data bundles (espeak-ng, open-jtalk).

## Project identity

Versta.Models is the Python toolchain that downloads, converts, quantizes and
bundles the AI models consumed by the Versta mobile app (sibling project
`Versta.Android`). Each top-level directory is a self-contained module that
produces tarball bundles deployable to the cloud object storage
(`models.versta.app`) from which the app side-loads models.

- The produced bundles must work fully offline in the Versta app: everything the model needs at runtime (weights, tokenizers, vocabularies, voices, metadata) must ship inside the bundle.
- Privacy-first: tooling fetches models only from their official upstream sources (Mozilla's translations bucket, Hugging Face). Never introduce third-party download redirects or telemetry.
- Source-available under the Source First License 1.1 (see `LICENSE.md`). Attribution and license notices must be preserved; never remove or alter licensing headers or notices.
- Python is the only primary language here. Every module is an independent [uv](https://docs.astral.sh/uv/) project: a PEP 621 `pyproject.toml` (hatchling backend) declares metadata and dependencies, and `uv sync` manages the module's virtual environment. There is no shared workspace — each module resolves and pins independently.

## Repo map

Each module directory is fully self-contained with its own `pyproject.toml`,
its own `versta/` Python package and its own uv-managed virtual environment:

- `translation/` — download and bundle Firefox (Bergamot) translation models. See `/translation/AGENTS.md`.
- `speech-recognition/` — export whisper.cpp ggml models (+ Silero-VAD) and bundle them. See `/speech-recognition/AGENTS.md`.
- `text-to-speech/` — convert Kokoro/Piper TTS models to ONNX/ORT and bundle them. See `/text-to-speech/AGENTS.md`.
- `object-character-recognition/` — export PaddleOCR models to ONNX/ORT, bundle them, and evaluate with OmniDocBench. See `/object-character-recognition/AGENTS.md`.
- `data/` — bundle shared runtime data (espeak-ng data, open-jtalk dictionary). See `/data/AGENTS.md`.
- `README.md` — per-module human-facing usage documentation; keep in sync. The root `README.md` stays a high-level overview and links out to these.

Per-module layout (repeated in every module):

- `versta/<tool>/__main__.py` — CLI entrypoint, invoked as `uv run python -m versta.<tool>` from the module directory.
- `versta/version.txt` (or `version.txt` at module root) — semantic version (`vX.Y.Z`) read at runtime and stamped into generated metadata. Also the source of the package version via hatchling's `[tool.hatch.version]` regex pattern (the `v` prefix is stripped for the PEP 440 distribution version).
- `versta/<tool>/metadata.py` — generates the `metadata.json` that ships inside each bundle and that the app parses at runtime.
- `models.json` (or `data.json` in `data/`) — the catalog definition consumed by the app for model downloads. Contains absolute URLs to `models.versta.app`.
- `pyproject.toml` — PEP 621 project definition: dependencies, `requires-python` and the dynamic package version read from `version.txt`.
- `.python-version` — the module's Python version (3.12 everywhere except `object-character-recognition/`, which needs 3.11 for the Paddle stack). uv selects the interpreter from this file.

Generated artifacts never live in git: `*/output`, `*/export`, `*/results`,
`*/tmp` and virtual environments are gitignored. Never commit model weights,
tarballs, checkpoints, caches or evaluation predictions.

## Python conventions

Apply these in every module; module-specific rules sit in the nested
AGENTS.md files.

- Python 3.11+/3.12, pinned per module by its `.python-version` file (uv selects the interpreter). `object-character-recognition/` requires 3.11 for the PaddlePaddle stack; the other modules use 3.12.
- **Strict typing, always.** Every function signature must be fully annotated — every parameter and every return type. No implicit `Any`, no untyped dicts for structured data: use `TypedDict` for dictionary-shaped values that cross module boundaries (see the existing `Output(TypedDict)` and `versta/*/typing.py` files). No bare `dict`/`list` annotations without type arguments — write `Dict[str, str]`, `List[dict]`, etc. Types must be correct and semantically meaningful: never paper over a mismatch with casts or `# type: ignore` just to silence a checker.
- Use `pathlib.Path` for all filesystem paths — never raw strings, never `os.path`. Accept `Path` parameters and return `Path` values.
- Google-style docstrings on all public functions: `Args:`, `Returns:`, `Raises:` sections with the type in parentheses after the parameter, matching the existing code.
- Ruff is the configured linter and formatter, shared via the root `ruff.toml` (Black-compatible defaults: 88-column, double quotes). `object-character-recognition/ruff.toml` extends it with a py311 target. Ruff runs inside each module's uv environment; check with `uv run ruff check` and `uv run ruff format`.
- Dependencies are declared per module in `pyproject.toml` (`[project] dependencies`), pinned as lower-bound/upper-bound ranges (`package>=x.y,<x.z`). Add new dependencies only to the module that needs them and with the same range style (`uv add` edits `pyproject.toml` directly). `uv.lock` files are gitignored by policy, so the ranges in `pyproject.toml` are the contract.
- Modules share code pattern, not code: each `versta/` package is self-contained. Do not import across module directories; copy small helpers into a module instead (the `bundle` packages already duplicate `bundle_tar.py`/`utils.py` per module — keep it that way).

## CLI pattern

Every tool is an argparse CLI in a `__main__.py`, run as a module from within
the module directory (`uv sync` installs the `versta` package editable, so
`uv run python -m versta.<tool>` works regardless of cwd):

```bash
cd <module>
uv sync
uv run python -m versta.<tool> --help
```

Conventions shared by all CLIs:

- Flag names use snake_case with dashes avoided (`--output_dir`, `--keep_intermediates`; note: `--model-type` in some newer modules — follow the local precedent).
- Boolean flags are `action="store_true"` with `default=False` unless the local precedent differs (translation's `bundle` uses `str2bool` for `--bidirectional`).
- `--output_dir` defaults to `Path("output")` and lands in the gitignored `*/output` tree.
- `--keep_intermediates` / `--keep_input` (and `--keep_downloads` in `data/`) control cleanup of intermediate and input artifacts; default is to clean up.
- Each `__main__.main()` is importable and callable directly (batch tools call the per-item tools' `main()`); keep argument parsing in `parse_args()` separate from logic in `main()`.

## Testing & verification

- There is currently no test suite and no CI. Changes are verified by running the CLIs end-to-end.
- The minimal smoke check for a module change is a successful run of the affected tool against a small model/language pair, writing into the gitignored `output/` tree.
- Do not add placeholder or template tests. If a test suite is introduced, tests must exercise live production code and assert behavior, not implementation details.
- Generated `metadata.json`/`models.json` files are machine-checked by the app; after changing a schema or field, verify against how `Versta.Android` parses them.

## Documentation

- The root `README.md` is a high-level overview of the repository (what it is, the module list) — keep it short. Module-specific usage (CLI invocations, pipelines, requirements, provenance) lives in each module's own `README.md`; do not duplicate it in the root file.
- Keep the relevant module `README.md` current. Update it in the same commit as any change to module capabilities, CLI arguments, supported models/architectures, or the produced bundle format.

## Git conventions

- Commit summaries are sentence-case and imperative, capitalized, with no trailing period. Examples: `Add module to export speech recognition models (Whisper)`, `Changing script to download and export Firefox models`.
- Keep commits small and focused; squash feature work before merge.
- Never commit generated artifacts: model weights (`.bin`, `.onnx`, `.ort`, `.safetensors`), tarballs, checksums, `output/`/`export/`/`results/` trees, `__pycache__`, virtual environments, or Hugging Face caches.
- `models.json` catalog files and `version.txt` files are committed; bump `version.txt` when the produced bundle format changes so the app can distinguish bundle generations.
