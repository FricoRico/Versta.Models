# pack-acceptance/

Inputs for the pack acceptance harness (NEU-58; runner: `versta.probe`,
workflow: `../../.github/workflows/pack-acceptance.yml`).

- `pins.json` — the probe inputs, pinned by content SHA-256 (`directions`)
  plus the Versta.Leanmt commit the `leanmt-probe` binary is built from.

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
# … (all four directions; see the workflow for the full loop)
uv run python -m versta.probe.pins --downloads_dir output
```
