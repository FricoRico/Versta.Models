# Versta.Models
This repository contains tooling to easily download, bundle and deploy AI models to be compatible with the Versta mobile app.

## Translation Models
The Versta mobile app uses translation models based on [Mozilla's Firefox Translations](https://github.com/mozilla/translations) models, powered by the [Bergamot](https://browser.mt/) (Marian) engine. These models are already shipped in the native on-device format (`.bin` / `.spm` files) and are conveniently split up in single direction language pairs (ie. English to Japanese). They are hosted on Mozilla's public Google Cloud Storage bucket and described by a [model registry](https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json).

Follow this guide to download the models and bundle them, which makes them compatible with the app.

### Downloading Firefox models
The models are downloaded directly from Mozilla's storage bucket. No conversion step is required, as the
models are already in the format expected by the Bergamot engine.

1. Install the required packages:
```bash
pip install -r requirements.txt
```
2. Run the download using the CLI:
```bash
python -m versta.download --src en --tgt es --architecture tiny --output_dir ./output
```

Replace `--src` and `--tgt` with the source and target language codes of choice (eg. `en` and `es`),
and `--architecture` with one of `tiny`, `base` or `base-memory` (defaults to `tiny`). After downloading,
you will have the model files in the `./output/en-es` directory, together with a `metadata.json` file.

### Bundling Models
After downloading the models, we need to bundle them to be used in the Android application. The models are side-loaded by the user during runtime. To make it convenient for the user to do so, we bundle the assets required for the models into a tarball. This can conveniently be done using the custom CLI tool.

1. Install the required packages:
```bash
pip install -r requirements.txt
```
2. Run the bundling using the CLI:
```bash
python -m versta.bundle --input_dir [$INPUT_DIR] --output_dir $OUTPUT_DIR
```

Replace `[$INPUT_DIR]` with each directory containing the downloaded models, separated by a space (eg. `en-es es-en`). After bundling, you will have a tarball in the same directory as the input directory. If no output directory is specified, the tarball will be saved in the `./output` directory.

By default we expect to deliver languages in pairs, so usually two or more input directories are expected. If you only want to support a single direction translation model, you can pass the optional argument `--language_pairs False` to the CLI.

### Example workflow
This is an example workflow to download the models and bundle them for the Android application. The models we will download are the English-Spanish pair in both directions.

1. Download the models:
```bash
python -m versta.download --src en --tgt es --architecture tiny --output_dir ./output
python -m versta.download --src es --tgt en --architecture tiny --output_dir ./output
```
2. Bundle the models:
```bash
python -m versta.bundle --input_dir ./output/en-es ./output/es-en --output_dir ./output
```

After running these commands, you will have a tarball in the `./output/en-es-bundle` directory containing the models.

### Batch workflow
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
python -m versta.batch --input_file pairs.json --output_dir ./output
```

This generates one tarball per language pair and a `models.json` definition that can be deployed to the cloud object storage.
