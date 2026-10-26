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
                                  │  • publish gate (NEU-60):│      └───────────────┘
                                  │    no passing scorecard  │
                                  │    in golden bands → no  │
                                  │    catalog written       │
                                  │  • generated catalog     │
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
registry's `uncompressedHash` (SHA-256 of the gunzipped payload). After hash
verification the model file must pass the leanmt handshake (see below) —
ingest fails closed on an incompatible embedded binary version. The upstream
per-model `metadata.json` provides the engine config (`enc-depth`, `dec-depth`,
`transformer-ffn-depth`, `transformer-heads`) and the model version. The
produced per-direction `metadata.json` is validated against the app contract
before the command exits — invalid output fails the run.

### `versta.bundle`

Combines directions (usually a mirrored pair) into `<langs>-bundle.tar.gz`
plus `<langs>-bundle.tar.sha256`. Steps: validate each input direction
(metadata + files on disk + leanmt handshake on the model payload) → validate
pair symmetry (when `--bidirectional`)
→ copy to intermediates → write bundle-level `metadata.json` (validated;
carries `leanmt_binary_version`) →
pack deterministically → validate the finished archive (including embedded
model headers vs. the declared version) → write checksum.

### `versta.batch`

Reads a checked-in-style catalog (`models.json`: list of language groups),
downloads + bundles each group, writes a generated `models.json` (validated),
then syncs computed fields (`size`, `bundle`, `checksum`, `score` ×100,
`version`) back into the input catalog in place (with a `.bak` backup) and
validates the updated catalog.

**Publish gate (NEU-60):** before any catalog write, every produced bundle
must be backed by a pass-rated pack-acceptance scorecard
(`versta.pack-scorecard/1`) covering its exact bytes (`bundle.file` +
`bundle.sha256`), with every direction loading, clearing the direction/chrF
sanity floor, at 100% digit preservation, and inside the pinned golden band
(`pack-acceptance/bands.json`, schema `versta.pack-bands/1`) vs. the
previously published bundle. Any failure refuses the run (non-zero exit, no
`models.json` written, input catalog untouched). `--skip_publish_gate` is
for development/dry-run exports only. Standalone check: `uv run python -m
versta.batch.publish_gate --bundle <tar>`. Thresholds, the re-pin procedure
and the failure taxonomy live in `PACK-ACCEPTANCE-RUNBOOK.md`. The scorecard
and bands shapes are manifest-adjacent: landed jointly with the pipeline
pair (NEU-38).

## Evaluation corpora (pinned)

Quality gates score against hash-pinned reference corpora, not live upstream
downloads. `evals/corpora/flores200-dev.json` pins the FLORES-200 dev snapshot
covering every shipped direction (artifact SHA-256, per-file hashes and
sentence counts); `uv run python -m versta.eval.corpora fetch` reproduces it
byte-identically into `output/corpora/` from recorded mirrors, verifying
before extraction and after. Provenance, license and the revision-bump
procedure: `evals/corpora/README.md`.

### Quality gate & ledger (`versta.eval`, `versta.eval.gate`, NEU-44/51/52)

A refresh ships only through the gate:

1. Score the candidate: `versta.eval` translates the hash-pinned FLORES-200
   snapshot (`evals/corpora/flores200-dev.json`) through the reference
   engine and writes a
   `versta.eval-result/1` JSON (scores, corpus pin, engine + decode
   provenance, artifact checksums, size). Line-misaligned engine output
   fails the run.
2. Gate it: `versta.eval.gate --result output/evals/<src>-<tgt>.json
   --record` compares the result against the direction's checked-in ledger
   (`evals/ledger/<direction>.json`, schema `versta.eval-ledger/1`) and
   appends the decision on success. chrF regression beyond the threshold
   (default 1.0, `versta/eval/config.py`) fails the run; overrides and
   corpus rebaselines ship only with a written `--reason`, recorded in the
   ledger entry.
3. Post the summary: every gate run generates the release eval-summary
   markdown (scores vs. previous version, corpus pin, engine version, size
   notes, decision + reason) — post the generated file, never handwritten
   prose.

Gate semantics, ledger fields and exit codes: `evals/README.md`.

CI wiring (NEU-53): PR CI runs the gate/metrics/ledger logic hermetically
against the tiny synthetic fixture pair (the suite refuses real network by
contract); the full pinned-corpus eval runs on release candidates only, via
the dispatchable `release-eval` workflow — see "CI wiring (NEU-53)" in
`evals/README.md`.

### `versta.eval` (quality measurement, NEU-44)

The quality stage: `uv run python -m versta.eval --source X --target Y
--bundle <bundle.tar.gz>` translates the pinned FLORES-200 dev snapshot
(checked-in manifest in `evals/corpora/`, hash-verified payloads fetched to
`output/corpora/` by `versta.eval.corpora fetch`) through the reference
runtime — upstream `bergamot-translator` as a black-box CLI for v1, the
LeanMT batch mode (NEU-50) once it lands — and scores sacreBLEU + chrF
(COMET-22 optionally, release tier only). The emitted result JSON
(schema `versta.eval-result/1`) records scores plus everything needed to
reproduce them: corpus revision and payload hashes, bundle/model
checksums, engine version and the fixed decode settings
(`versta/eval/config.py`, mirroring Firefox production). Re-running with
identical inputs reproduces identical scores. See `evals/README.md` for
the manifest and result contracts, the release gate (`versta.eval.gate`,
NEU-52) and the `evals/` ledger it records into, and the CI wiring
(NEU-53). Tests are hermetic (a fake engine over a tiny synthetic corpus);
scoring a real model needs the corpus payloads fetched and a bergamot
CLI available.

## The app contract (kept in `versta/schema.py`)

| Artifact | App type (`core/entity`) | Required fields |
|---|---|---|
| per-direction `metadata.json` | `LanguageModelMetadata` | `base_model`, `source_language`, `target_language`, `files{model, vocabulary, shortlist}` (+ `target_vocabulary`: optional for shared-`Wemb` models, required for split-Wemb two-vocabulary models — see below), `config{encoder_layers, decoder_layers, num_heads}` > 0 (+ optional `ffn_depth=2`, `split_mode="sentence"`); referenced files must exist |
| bundle `metadata.json` | `LanguageBundleMetadata` | `version`, `leanmt_binary_version` (see handshake below), `languages` (non-empty; even count when bidirectional), `bidirectional`, `metadata[]{directory, source_language, target_language}` |
| checked-in catalog | `ExternalLanguageModelDefinition` | `base_model`, `source_language`, `target_language`, `bidirectional`, `architectures` ∈ {`MarianMTModel`,`BergamotTinyModel`}, `score`, `size` > 0, `version`, `bundle`, `checksum` (absolute URIs); bidirectional groups are exactly two mirrored entries agreeing on size/version/URLs |

Extend additively only — never rename or drop fields. Format changes go
through Elin Sørensen (LeanMT); loader questions go through Maya Lindqvist.

### Language codes at the bundle boundary (NEU-133)

There are two code layers, and the mapping between them is owned, singularly,
by `versta/language_codes.py`:

- **Catalog layer** (checked-in `models.json`): BCP-47-style display codes —
  `zh-Hans` for simplified Chinese. These are the human-facing fields the app
  lists packs by.
- **Bundle-boundary layer** (direction `metadata.json`, bundle
  `metadata.json.languages`, `<slug>-bundle.tar.gz` archive names): the
  Mozilla registry's pair keys — bare ISO-639 codes, with Chinese as `zh`
  (simplified) / `zh_hant` (traditional). This layer is canonical at the
  artifact boundary on purpose: bundle payloads are derived 1:1 from Mozilla's
  published artifacts, archive names stay derivable from contents, and probe
  corpora (`en-zh`/`zh-en`) measure the packs exactly as shipped. Re-coding
  bundles to catalog codes would re-burn every published hash without changing
  a single model byte.

Contract rules, all fail-fast enforced:

1. Bundles only ever contain registry-normalized codes — `versta.bundle`
   refuses a direction whose `metadata.json` declares a BCP-47 code
   (`assert_boundary_codes`), so `zh-Hans` can never leak into a manifest or
   archive name.
2. The catalog↔bundle join is **never string equality**: both sides are
   compared through `normalize_language` (this is what `versta.batch`'s
   catalog sync does via `_entry_key`, and what the app must do when joining
   catalog language codes to bundle-declared languages at install time).
3. Every catalog entry's `bundle`/`checksum` URL basename must be the
   deterministic slug of its group's normalized codes
   (`validate_catalog_bundle_alignment`, covered by the checked-in catalog
   test): catalog `zh-Hans` ↔ bundle `en-zh-bundle.tar.gz` is valid, catalog
   pointing at `en-zh-Hans-bundle.tar.gz` fails CI.

## leanmt handshake (`leanmt_binary_version`, NEU-25)

The app loads the model `.bin` through leanmt, which fails fast when the
file's embedded binary format version (a little-endian uint64, the first field
of the bergamot layout) differs from its `kBinaryFileVersion` (see
Versta.Leanmt `src/Io.hh`, and "Model loading contract" in its README). That
error only surfaces after JNI. So bundles additionally **declare** the version
in the bundle-level `metadata.json`:

```json
{ "version": "v2.1.0", "leanmt_binary_version": 1, ... }
```

The app pre-flights this field before calling into native code and shows a
clean error on stale bundles instead of a native failure. The field name and
semantics are the negotiated contract with Elin Sørensen.

Converter-side, the pipeline **fails closed** at three points
(`versta/leanmt.py`, mirrored in `versta/schema.py`):

1. `versta.download` aborts when a freshly downloaded model's embedded header
   version ≠ `LEANMT_BINARY_VERSION` (1);
2. `versta.bundle` re-checks every input direction's model payload before
   packing, and `validate_bundle_metadata` refuses manifests that miss the
   field or declare anything but `1`;
3. `validate_bundle_archive` compares each archived model's embedded header
   against the declared value, so a hand-edited manifest cannot smuggle an
   incompatible model through.

A future format bump is a coordinated change: leanmt raises
`kBinaryFileVersion`, this repo raises `LEANMT_BINARY_VERSION`, and the app
gates old bundles via pre-flight. App builds parse bundle metadata with
`ignoreUnknownKeys = true`, so the new key is tolerated by app versions that
pre-date the pre-flight logic.

### Two-vocabulary models (`files.target_vocabulary`, NEU-26)

Firefox MT pairs with disjoint source/target vocabularies (ja, zh) ship two
sentencepiece files and their models carry split `encoder_Wemb`/
`decoder_Wemb` embedding tensors on disk; shared-vocabulary pairs carry a
single `Wemb` instead. leanmt fails fast when a split-Wemb model loads with
an empty target vocabulary — silently decoding garbage was the pre-NEU-26
bug. The converter enforces the same rule at the same signal, walking the
tensor table exactly as `load_items` does (version → header count → headers
→ NUL-terminated names, bounds-checked):

- `validate_language_model_files` (runs after `versta.download` and on every
  `versta.bundle` input direction) requires `files.target_vocabulary` —
  present, non-blank and on disk — when the model's tensor table contains
  `encoder_Wemb`/`decoder_Wemb`; for shared-`Wemb` models the field stays
  optional.
- `validate_bundle_archive` rechecks the rule against the payload inside the
  produced tarball, after the binary-version comparison, so a hand-edited
  manifest cannot decouple manifest and payload.

A registry entry missing `trgVocab` for a split-Wemb model aborts the export
with this error — that is Mozilla-side breakage to raise upstream, never
something to paper over.

Generation note (ja): the registry's en-ja/ja-en `base-memory` entries are
the generation Firefox ships as remote-settings model v2.3 (byte-identical;
en-ja model sha-256 `59ae659f…`), with `dec-depth: 4` in `modelConfig` —
the `dec-depth: 2` `base` entries are the older cjk_split_vocabs generation;
they only ship when `--architecture base` is requested explicitly. The
architecture preference (`tiny` → `base-memory` → `base`) keeps unattended
batch exports on the current generation.

## leanmt translate CLI (`versta.engine-cli/1`, NEU-50)

The quality harness scores the *shipped* engine against the *shipped*
bundle by driving leanmt's bench/translate binary as a black-box CLI. The
engine side of that contract is `leanmt-bench translate` (Versta.Leanmt,
`bench/bench_translate.cc`; full semantics in its README "Batch translation
(file → file)"); this repo pins the contract version as
`LEANMT_CLI_SPEC = "versta.engine-cli/1"` in `versta/leanmt.py` — same
handshake style as `leanmt_binary_version` above. The harness must refuse
to run against a binary whose `--version` does not report the pinned spec
identifier; a spec bump is a deliberate, coordinated event that makes
old/new scores non-comparable by construction.

```bash
leanmt-bench translate \
    --model model.bin --vocab vocab.src.spm [--target-vocab vocab.trg.spm] \
    --shortlist lex.bin \
    --input input.txt --output output.txt \
    [--threads N] [--beam N] [--batch N]
```

Contract summary (the engine README is normative):

- `--input` holds UTF-8 sentences, one per line; `--output` receives exactly
  one translation per input line, in order, each `\n`-terminated. Blank
  input lines abort with the line number (matches the corpus rule "no blank
  lines; output must stay line-aligned"); a translation containing a raw
  newline aborts the run rather than corrupting alignment.
- Exit codes: `0` success, `1` runtime failure (unreadable input,
  model-load error including the NEU-25/NEU-26 fail-fasts, decode failure,
  unwritable output), `2` usage error.
- Deterministic for a fixed model + decode settings: bit-identical across
  reruns, thread counts and `--batch` values; the translation cache is off
  in this mode. `--threads`, `--beam` (default 3, `1` = greedy) pin the
  decode; the effective settings are echoed to stderr and `--version`
  reports engine + spec versions for the result JSON's `engine` block.
- Split-Wemb pairs (NEU-26) pass `--target-vocab`, mirroring the bundle's
  `files.target_vocabulary`.

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

# Many pairs from a catalog (publishes only with passing scorecards; use
# --skip_publish_gate for a development/dry-run export):
uv run python -m versta.batch --input_file models.json --output_dir ./output/export

# Is this bundle publishable yet? (exit 0 publishable / 2 refused):
uv run python -m versta.batch.publish_gate --bundle output/en-es-bundle.tar.gz

# Golden bands: verify the committed pin against the committed scorecards,
# or re-pin (dedicated commit only — see PACK-ACCEPTANCE-RUNBOOK.md):
uv run python -m versta.probe.bands --check

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
driven against a mocked Mozilla registry (hermetic, no network) with the
publish gate's refuse/pass paths, plus `versta.probe.bands --check` locking
the pinned golden bands to the committed scorecards. A real-model
export is validated manually per release (see README's verification section).

`.github/workflows/pack-acceptance.yml` sweeps every published bundle on
arm64 per commit (probe + adversarial negative control + publish-gate
demonstration: the deliberately mislabeled control pack must be refused, a
proven bundle must pass). `.github/workflows/pack-sweep-weekly.yml` runs the
same sweep on Mondays and files one deduped GitHub issue per regressed
bundle (label `pack-acceptance`); see `PACK-ACCEPTANCE-RUNBOOK.md`.
