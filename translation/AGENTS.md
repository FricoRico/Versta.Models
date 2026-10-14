# translation/

Firefox (Bergamot/Marian) translation models for the Versta app. Read the
root `/AGENTS.md` first; it defines the shared Python, typing and CLI
conventions this module follows.

## Module purpose

This module downloads translation models published by
[Mozilla's Firefox Translations project](https://github.com/mozilla/translations)
from Mozilla's public Google Cloud Storage bucket and bundles them into the
tarballs the Versta app side-loads at runtime.

- Models are already in the native on-device Bergamot format (`.bin` model, `.spm` vocabulary, lexical shortlist). There is **no conversion step** — download, verify, bundle.
- Models are single-direction (e.g. `en` → `es`). The app treats a language pair as one bundle containing both directions.
- Dependency footprint is deliberately tiny: only `requests` (plus stdlib `urllib`/gzip/hashlib for the actual download path). Keep it that way.

## Tools

All tools run from within this directory after `uv sync` (dependencies are declared in `pyproject.toml`).

### `versta.download`

Downloads one language direction from the Mozilla registry and writes it with
a `metadata.json` into `output/<src>-<tgt>/`:

```bash
uv run python -m versta.download --source en --target es --architecture tiny --output_dir ./output
```

- `--source`/`--target` (required): language codes. Inputs are normalized before registry lookup (`normalize_language`): region subtags are stripped (`en-US` → `en`) and Chinese script tags map to `zh` (simplified) / `zh_hant` (traditional). Never normalize `no`/`nb`/`nn`/`hbs`.
- `--architecture`: one of `tiny` (default), `base`, `base-memory`. When omitted in batch entries, the best available architecture is picked with preference `tiny` → `base-memory` → `base`.
- `--registry_url`: defaults to Mozilla's production `models.json` registry; the registry's `baseUrl` is used as the download base.
- Downloads are gzip-decompressed on the fly and verified against the registry's `uncompressedHash` (SHA-256); a hash mismatch aborts with an error.

### `versta.bundle`

Bundles one or more downloaded directions into a single tarball plus a
`.sha256` checksum:

```bash
uv run python -m versta.bundle --input_dir ./output/en-es ./output/es-en --output_dir ./output
```

- `--input_dir` (one or more): directories produced by `versta.download`.
- `--bidirectional` (`str2bool`, default `True`): validates that the inputs form a proper two-direction pair. Set to `False` for a single-direction bundle.
- `--keep_intermediates` / `--keep_input` (`store_true`, default off): keep the staging directory and the input downloads, respectively.
- Output: `<languages>-bundle.tar.gz` and `<languages>-bundle.tar.sha256`, where `<languages>` is the joined unique language codes (e.g. `en-es`). Bundle metadata comes from `bundle/metadata.py` (`generate_metadata`) and is stamped with the module `version.txt`.

### `versta.batch`

Batch download + bundle for many pairs at once from a JSON input file:

```bash
uv run python -m versta.batch --input_file models.json --output_dir ./output/export
```

- `--input_file`: JSON list of pairs, each pair a list of `{"source_language", "target_language", "architecture"}` entries.
- `--link_prefix`: URL prefix written into the generated catalog (default `https://models.versta.app/translation/`).
- Output: one tarball per pair plus a deployable `models.json` catalog; the input file is updated in place (with `.bak` backup) so the checked-in `models.json` stays in sync with what was actually exported.

### `versta.eval`

The quality harness (NEU-44/NEU-51): translates the pinned FLORES-200
snapshot for one direction through the reference runtime and scores
sacreBLEU + chrF, writing a versioned result JSON:

```bash
uv run python -m versta.eval --source en --target es --bundle output/en-es-bundle.tar.gz
# or, before bundling:
uv run python -m versta.eval --source en --target es --direction_dir output/en-es
```

- One command, one direction, one result JSON (schema `versta.eval-result/1`): scores, corpus revision, engine version + fixed decode settings, artifact checksums. The release gate and the `evals/` ledger (NEU-52) consume it; both contracts are documented in `evals/README.md`.
- The corpus is the pinned snapshot in `evals/` (`corpora.json` + payloads), verified by hash before any engine time is spent.
- The engine is a black-box CLI — no engine code lands in this repo. v1 runtime is upstream `bergamot-translator` via `--bergamot_bin` or `--bergamot_image` (env: `VERSTA_BERGAMOT_BIN` / `VERSTA_BERGAMOT_IMAGE`), with decode settings fixed to the production Firefox values in `versta/eval/config.py`. `--engine leanmt` is reserved for the LeanMT batch CLI (NEU-50).
- Output must stay line-aligned with input; misalignment fails the run instead of scoring garbage.
- `--comet` adds release-tier COMET-22 (optional `unbabel-comet` dependency, cached weights — never required on the PR path).
- `sacrebleu` is the harness's one runtime dependency; the rest of the module keeps its zero-dependency footprint.

## Conventions specific to this module

- Language codes are the registry's bare ISO-639 keys; bundle names, metadata fields and directory names all use the `<src>-<tgt>` form.
- `version.txt` lives at `versta/version.txt` and is read once at import time in each `__main__.py`. Bump it when the bundle format or metadata schema changes — the app uses it to distinguish bundle generations (registry currently at `v2.1.0`, which added `leanmt_binary_version`).
- The per-direction `metadata.json` (written by `versta/download/download.py`) records: `directory`, `source_language`, `target_language`, `architecture`, `base_model`, `score` (COMET-22 from flores200-plus metrics), `version`, `files` (model/vocabulary/shortlist filenames) and `config` (encoder/decoder layers, ffn depth, heads, split mode = `sentence`). These fields are consumed by the app and by `versta.bundle`; extend additively, never rename or drop.
- `versta/schema.py` is the runtime contract gate: it mirrors the app's Kotlin loader shapes (`LanguageModelMetadata`, `LanguageBundleMetadata`, `ExternalLanguageModelDefinition`) and is enforced fail-fast in download, bundle and batch. When the app contract changes, update `versta/schema.py` and its tests in the same change.
- `versta/leanmt.py` is the format handshake with the leanmt loader (NEU-25): `LEANMT_BINARY_VERSION` mirrors `kBinaryFileVersion` in Versta.Leanmt (`src/Io.hh`). Bundles declare `leanmt_binary_version` in the bundle-level `metadata.json` so the app can pre-flight before JNI, and the pipeline fails closed — download, bundle and archive validation all reject embedded/declared versions other than 1. Never hard-code the number elsewhere; import the constant. Format bumps are negotiated with Elin Sørensen. The module also walks the binary's tensor table exactly like the loader's `load_items` (bounds-checked, fail-closed on foreign versions): models with split `encoder_Wemb`/`decoder_Wemb` embeddings (two-vocabulary pairs like ja/zh) must declare `files.target_vocabulary` — leanmt fails fast without it (NEU-26). See PIPELINE.md "Two-vocabulary models".
- `versta/batch/typing.py` defines the `ModelFile`/`ExportedModel`/`ExportedBundle` TypedDicts crossing the batch boundary — keep them fully typed and in sync with what `versta.bundle.main()` actually returns.
- Never fetch models from anywhere other than the Mozilla registry/bucket.

## Verification

The test suite lives in `tests/` and runs in CI (`.github/workflows/ci.yml`)
alongside ruff and mypy:

```bash
uv sync
uv run pytest -v                    # includes a hermetic smoke export (mocked registry)
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

For changes touching produced artifacts, additionally smoke-check the real
path end-to-end against Mozilla's bucket (small pair, architecture `tiny`) and
confirm the tarball, checksum and `metadata.json` content:

```bash
uv run python -m versta.download --source en --target es --architecture tiny
uv run python -m versta.download --source es --target en --architecture tiny
uv run python -m versta.bundle --input_dir output/en-es output/es-en --keep_intermediates --keep_input
# reproducibility: rerun from a clean tree and compare the .sha256 outputs
```

See `PIPELINE.md` for the pipeline architecture and the app contract enforced
by `versta/schema.py`.
