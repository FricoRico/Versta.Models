# Translation Models
The Versta mobile app uses translation models based on [Mozilla's Firefox Translations](https://github.com/mozilla/translations) models, powered by the [Bergamot](https://browser.mt/) (Marian) engine. These models are already shipped in the native on-device format (`.bin` / `.spm` files) and are conveniently split up in single direction language pairs (ie. English to Japanese). They are hosted on Mozilla's public Google Cloud Storage bucket and described by a [model registry](https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json).

Follow this guide to download the models and bundle them, which makes them compatible with the app.

## Downloading Firefox models
The models are downloaded directly from Mozilla's storage bucket. No conversion step is required, as the
models are already in the format expected by the Bergamot engine.

1. Install the required packages (requires [uv](https://docs.astral.sh/uv/)):
```bash
uv sync
```
2. Run the download using the CLI:
```bash
uv run python -m versta.download --source en --target es --architecture tiny --output_dir ./output
```

Replace `--source` and `--target` with the source and target language codes of choice (eg. `en` and `es`),
and `--architecture` with one of `tiny`, `base` or `base-memory` (defaults to `tiny`). After downloading,
you will have the model files in the `./output/en-es` directory, together with a `metadata.json` file.

## Bundling Models
After downloading the models, we need to bundle them to be used in the Android application. The models are side-loaded by the user during runtime. To make it convenient for the user to do so, we bundle the assets required for the models into a tarball. This can conveniently be done using the custom CLI tool.

1. Install the required packages:
```bash
uv sync
```
2. Run the bundling using the CLI:
```bash
uv run python -m versta.bundle --input_dir [$INPUT_DIR] --output_dir $OUTPUT_DIR
```

Replace `[$INPUT_DIR]` with each directory containing the downloaded models, separated by a space (eg. `en-es es-en`). After bundling, you will have a tarball in the same directory as the input directory. If no output directory is specified, the tarball will be saved in the `./output` directory.

By default we expect to deliver languages in pairs, so usually two or more input directories are expected. If you only want to support a single direction translation model, you can pass the optional argument `--bidirectional False` to the CLI.

## Example workflow
This is an example workflow to download the models and bundle them for the Android application. The models we will download are the English-Spanish pair in both directions.

1. Download the models:
```bash
uv run python -m versta.download --source en --target es --architecture tiny --output_dir ./output
uv run python -m versta.download --source es --target en --architecture tiny --output_dir ./output
```
2. Bundle the models:
```bash
uv run python -m versta.bundle --input_dir ./output/en-es ./output/es-en --output_dir ./output
```

After running these commands, you will have a tarball in the `./output/en-es-bundle` directory containing the models.

## Batch workflow
To download and bundle many language pairs at once, provide a JSON file describing the pairs to the batch CLI:

```json
[
  [
    { "source_language": "en", "target_language": "es", "architecture": "tiny" },
    { "source_language": "es", "target_language": "en", "architecture": "tiny" }
  ]
]
```

```bash
uv run python -m versta.batch --input_file pairs.json --output_dir ./output
```

This generates one tarball per language pair and a `models.json` definition that can be deployed to the cloud object storage.

## Verification

Bundles are produced deterministically: identical inputs yield byte-identical
`.tar.gz` output, so exports can be verified by comparing SHA-256 checksums
between runs or machines (set `SOURCE_DATE_EPOCH` to pin a non-default
timestamp; see `PIPELINE.md`).

Produced artifacts are validated fail-fast against the app's loader schema
(`versta/schema.py`): per-direction `metadata.json`, the bundle-level
metadata, the archive contents and the generated catalog.

The test suite (including a hermetic smoke export against a mocked registry)
and the linters run in CI (`.github/workflows/ci.yml`):

```bash
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

See `PIPELINE.md` for the pipeline architecture, the app contract and the
reproducibility guarantees.

## Pack acceptance probing (NEU-60)

A bundle is not proven by passing the shape gates above — it is proven when
it *loads and translates*. `versta.probe` is the pack acceptance runner:
checksum → the same metadata/parse gate the app goes through
(`versta/schema.py`) → loads each declared direction through `leanmt-probe`
(the pinned Versta.Leanmt host binary, built with the exact
`leanmt::Package`/`Model::Config` mapping the app's JNI bridge uses, with
app-default decode) → translates the packaged traveler probe corpus
(`versta/probe/corpora/`, 40 sentences per direction covering numbers, times,
prices, addresses, directions, emergencies) → writes one scorecard JSON per
bundle to `scorecards/`.

```bash
uv run python -m versta.probe \
    --bundle output/en-es-bundle.tar.gz \
    --probe_bin <path-to-leanmt-probe> \
    --output_dir output/probe \
    --expect pass            # '--expect direction-fail' for the adversarial control
```

Exit 0 iff the outcome matches `--expect`. The v1 gates are load and the
chrF direction floor (the NEU-19 mislabeled-direction class); digit
preservation (NEU-30), p50/p95 latency and peak RSS are measured and
published on every scorecard and become publish gates in NEU-62 once bands
are calibrated from the first full sweep.

The adversarial negative control is built from a real payload with swapped
direction claims:

```bash
uv run python -m versta.probe.adversarial --model_dir output/en-es --output_dir output/adversarial
```

on the `ubuntu-24.04-arm` CI runner this all runs end to end on every commit
(`.github/workflows/pack-acceptance.yml`), against model contents pinned in
`pack-acceptance/pins.json` (content SHA-256s, bench-baseline re-pinning
discipline). Committed scorecards for the first sweep live in `scorecards/`
(`en-es`, `en-ja`, `es-en.adversarial`).
