# Translation pipeline — architecture & runbook

How a Firefox Translations (Bergamot/Marian) model becomes the on-device
package the Versta app ships. Mozilla publishes models already in the native
on-device format (int8 `.bin` weights, sentencepiece `.spm` vocabulary, lexical
shortlist); this pipeline **downloads, verifies, and bundles** — it never
re-quantizes or re-compiles weights.

## Stages

```
Mozilla GCS bucket                      Versta.Models                    models.versta.app
┌────────────────────────┐        ┌──────────────────────────┐         ┌───────────────┐
│ registry models.json   │        │ versta.download          │         │               │
│  (entries per pair &   │───────▶│  • resolve pair + arch   │         │               │
│   arch, + hashes)      │        │  • download .gz files    │         │               │
│ model.bin.gz           │        │  • gunzip + SHA-256 check│         │               │
│ lex.50.es.spm.gz …     │        │  • fetch upstream        │         │               │
│ upstream metadata.json │        │    metadata.json         │         │               │
└────────────────────────┘        │  ⇒ write per-direction   │         │               │
                                  │    metadata.json         │         │               │
                                  │  ⇒ validate vs app schema│         │               │
                                  └──────────┬───────────────┘         │               │
                                             ▼                          │               │
                                  ┌──────────────────────────┐          │               │
                                  │ versta.bundle            │          │               │
                                  │  • validate inputs       │          │               │
                                  │  • check mirrored pairs  │          │               │
                                  │  • deterministic tar.gz  │          │               │
                                  │  • bundle metadata.json  │          │               │
                                  │  ⇒ validate archive vs   │          │               │
                                  │    app schema            │          │               │
                                  └──────────┬───────────────┘          │               │
                                             ▼                          ▼               │
                                  ┌──────────────────────────┐      deploy (manual)
                                  │ versta.batch             │      │ tarball + .sha256 │
                                  │  • loop pairs            │      │ catalog models.json│
                                  │  • generated catalog     │      └───────────────┘
                                  │  • sync checked-in       │
                                  │    models.json in place  │
                                  │  ⇒ validate both         │
                                  └──────────────────────────┘
```

### `versta.download`

One direction at a time (`--source en --target es --architecture tiny`).
Language codes are normalized (`en-US`→`en`, `zh-Hant`→`zh_hant`;
`no`/`nb`/`nn`/`hbs` never collapsed). Architecture selection prefers
`tiny` → `base-memory` → `base`. Every downloaded file is verified against the
registry's `uncompressedHash` (SHA-256 of the gunzipped payload). The upstream
per-model `metadata.json` provides the engine config (`enc-depth`, `dec-depth`,
`transformer-ffn-depth`, `transformer-heads`) and the model version. The
produced per-direction `metadata.json` is validated against the app contract
before the command exits — invalid output fails the run.

### `versta.bundle`

Combines directions (usually a mirrored pair) into `<langs>-bundle.tar.gz`
plus `<langs>-bundle.tar.sha256`. Steps: validate each input direction
(metadata + files on disk) → validate pair symmetry (when `--bidirectional`)
→ copy to intermediates → write bundle-level `metadata.json` (validated) →
pack deterministically → validate the finished archive → write checksum.

### `versta.batch`

Reads a checked-in-style catalog (`models.json`: list of language groups),
downloads + bundles each group, writes a generated `models.json` (validated),
then syncs computed fields (`size`, `bundle`, `checksum`, `score` ×100,
`version`) back into the input catalog in place (with a `.bak` backup) and
validates the updated catalog.

## The app contract (kept in `versta/schema.py`)

| Artifact | App type (`core/entity`) | Required fields |
|---|---|---|
| per-direction `metadata.json` | `LanguageModelMetadata` | `base_model`, `source_language`, `target_language`, `files{model, vocabulary, shortlist}` (+ optional `target_vocabulary`), `config{encoder_layers, decoder_layers, num_heads}` > 0 (+ optional `ffn_depth=2`, `split_mode="sentence"`); referenced files must exist |
| bundle `metadata.json` | `LanguageBundleMetadata` | `version`, `languages` (non-empty; even count when bidirectional), `bidirectional`, `metadata[]{directory, source_language, target_language}` |
| checked-in catalog | `ExternalLanguageModelDefinition` | `base_model`, `source_language`, `target_language`, `bidirectional`, `architectures` ∈ {`MarianMTModel`,`BergamotTinyModel`}, `score`, `size` > 0, `version`, `bundle`, `checksum` (absolute URIs); bidirectional groups are exactly two mirrored entries agreeing on size/version/URLs |

Extend additively only — never rename or drop fields. Format changes go
through Elin Sørensen (LeanMT); loader questions go through Maya Lindqvist.

## Reproducibility contract

Identical inputs must produce byte-identical bundles:

- Archive members are sorted by archive path; uid/gid are zeroed, names
  blanked, modes normalized (0644/0755), and every mtime is the fixed epoch
  `0` (`SOURCE_DATE_EPOCH` overrides it when a release timestamp is wanted).
- `versta.download` output depends only on the registry entry contents: file
  payloads are hash-verified, and generated `metadata.json` is stable for a
  given upstream version.
- Verify with: `pytest -k reproducible` (offline) or two consecutive exports
  of a pair and `diff` their `.sha256` files.

## Running it (new engineer)

```bash
cd translation
uv sync                                          # installs pinned toolchain (uv-managed)

# One pair, end to end:
uv run python -m versta.download --source en --target es --architecture tiny --output_dir ./output
uv run python -m versta.download --source es --target en --architecture tiny --output_dir ./output
uv run python -m versta.bundle --input_dir ./output/en-es ./output/es-en --output_dir ./output

# Many pairs from a catalog:
uv run python -m versta.batch --input_file models.json --output_dir ./output/export

# Quality gates (what CI runs):
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -v                                 # includes the offline smoke export
```

Artifacts land in the gitignored `output/` tree. Deploying to
`models.versta.app` is a manual step outside the tools (see the checked-in
`models.json` for the published catalog the app consumes).

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and PRs touching
`translation/`: ruff lint + format check, mypy on the `versta` package, and
the pytest suite — including `tests/test_smoke_export.py`, a full batch export
driven against a mocked Mozilla registry (hermetic, no network). A real-model
export is validated manually per release (see README's verification section).
