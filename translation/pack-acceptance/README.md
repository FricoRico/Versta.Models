# pack-acceptance/

Inputs for the pack acceptance harness (NEU-58–NEU-60; runner: `versta.probe`,
workflows: `../../.github/workflows/pack-acceptance.yml` (+ weekly:
`pack-sweep-weekly.yml`); ops: `../PACK-ACCEPTANCE-RUNBOOK.md`).

- `pins.json` — the probe inputs, pinned by content SHA-256 (`directions`)
  plus the Versta.Leanmt commit the `leanmt-probe` binary is built from.
- `bands.json` — the golden-output bands (schema `versta.pack-bands/1`,
  NEU-60): per shipped direction the chrF measured on the previously
  published bundle, the publish-allowed drop below it, and the measurement's
  provenance. Derived from `../scorecards/` by `python -m versta.probe.bands`
  (`--check` verifies; CI enforces tree consistency).

## Re-pinning discipline

Same rule as the leanmt bench baselines (NEU-16/22/23): pins move only in a
dedicated commit, with the measured scorecard diff attached to the PR. A pin
mismatch in CI means upstream (Mozilla's registry) re-published underneath
us or someone edited pins casually — investigate first, never bump to make
CI green.

When the leanmt pin moves to a commit that changes probe output, regenerate
`../scorecards/` on the arm64 CI runner in the same commit and include the
diff in the PR.

## Verifying locally

```bash
cd translation
uv sync
uv run python -m versta.download --source en --target es --architecture base-memory
# … (all four directions; use --architecture tiny for ja-en — the v2.0.0
# catalog ships the tiny export for that direction, see pins.json; see the
# workflow for the full loop)
uv run python -m versta.probe.pins --downloads_dir output
```
