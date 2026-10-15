# text-to-speech

Voice-model conversion and bundling for the Versta app.

**Current ship:** Supertonic 3 converted to MNN int8 (`supertonic3` bundle,
v2.0.0). **Legacy:** Piper voices converted to ONNX/ORT. The Kokoro exporter
is archived in `versta/export/_archive/` (superseded by Supertonic 3;
bundle removal is staged with the 5/6 workstream).

## Supertonic 3 → MNN (versta.supertonic3)

Pipeline: pinned upstream download → sha256-verified passthroughs →
MNNConvert per stage (fp32 control, fp16 gate reference, int8 shipped) →
seeded numerics verification → lint → deterministic tarball + sha256 →
feed entry → ledger append.

```bash
uv sync
git submodule update --init vendor/mnn   # MNN 3.6.1, provides MNNConvert

# 1. Convert (fails closed on any upstream content drift)
uv run python -m versta.supertonic3 convert --upstream_revision <commit-sha>

# 2. Verify numerics vs the ONNX fp32 reference (seeded; int8 must gate-pass)
uv run python -m versta.supertonic3 verify \
  --snapshot <snapshot dir from step 1> --converted output/supertonic3 \
  --upstream_revision <commit-sha> --output output/supertonic3/verification.json

# 3. Build the shippable bundle (int8-only; ≤120 MB cap enforced)
uv run python -m versta.supertonic3 bundle --converted output/supertonic3 --version v2.0.0

# 4. Record the release in the checked-in conversion ledger (refuses a
#    failed gate without an explicit --override_reason)
uv run python -m versta.supertonic3 record --converted output/supertonic3 \
  --bundle output/supertonic3-bundle.tar.gz --version v2.0.0

# 5. Merge the feed entry (models.json) with the measured tarball size
uv run python -m versta.supertonic3 feed --version v2.0.0 \
  --bundle output/supertonic3-bundle.tar.gz
```

- **Precisions:** int8 ships. fp16/fp32 exist only as quality-gate
  references (NEU-64 decision 4; the fp16 set is 191 MB and breaks the
  120 MB budget, and per-language precision is impossible with shared
  weights).
- **Determinism:** probe inputs are seeded (`--seed`, default 42) and every
  MNN run is repeated bit-exact; the conversion/engine RNG split is a
  contract item with the native engine (seeded `std::mt19937` on the device
  side, 3/6).
- **Verification gates** live in `versta/supertonic3/config.py`,
  calibrated to the NEU-64 spike table (int8 LSD vs fp32 ≈ 7.5–15 dB is
  expected divergence, not failure; fp16 rows are informational only).
- **OpenRAIL:** the bundle carries the upstream BigScience Open RAIL-M
  `LICENSE`, a generated README with the §III.4 modification notice, and
  attribution; the bundler lints these into existence.
- **Ledger:** each shipped version appends one entry to
  `evals/ledger/supertonic3.json` (schema `versta.tts-conversion-ledger/1`)
  with source revision, toolchain, artifact hashes and the verification
  report — append-only, atomic writes, same conventions as the
  versta.eval ledger.

## Piper → ONNX/ORT (versta.export / versta.bundle / versta.batch)

Unchanged legacy path; see `AGENTS.md`. `--model_format` accepts `piper`
only; `kokoro` raises with a pointer to the archive.

## Tests

```bash
uv run pytest tests/
```
