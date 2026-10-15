"""
Supertonic 3 → MNN conversion pipeline (NEU-68).

Converts the upstream `Supertone/supertonic-3` ONNX model set
(duration_predictor, text_encoder, vector_estimator, vocoder) into the int8
MNN bundle the Versta app ships, with the fp16 set produced as the
quality-gate reference (NEU-64 decision: fp16 is verified against, never
co-shipped).

Subcommands (see `python -m versta.supertonic3 --help`):

- ``convert`` — download the pinned upstream revision, verify the
  byte-identical passthroughs by sha256, and run MNNConvert per stage
  (fp32 + fp16 + int8-weight).
- ``verify`` — seeded numerics comparison of the MNN outputs against the
  ONNX reference (SNR/cosine per stage, LSD/spectral-r on wav) against the
  threshold table calibrated on the NEU-64 spike data.
- ``bundle`` — assemble `supertonic3-bundle.tar.gz` (+ `.sha256`) with
  metadata.json, LICENSE, generated README/NOTICE (OpenRAIL obligations),
  and the bundle-lint gate (required files, ≤ 120 MB int8 cap).
- ``feed`` — generate/merge the `supertonic3` entry of `models.json`
  (62 voice rows, canonical order pinned by the app-side feed test).

Every run appends to the checked-in conversion ledger under
``evals/ledger/supertonic3.json`` (schema ``versta.tts-conversion-ledger/1``),
following the append-only/atomic conventions of the versta.eval ledger.
"""
