# Pack acceptance runbook

Operational companion to `PIPELINE.md` for the pack acceptance harness
(epic NEU-56; proposal NEU-45 rev 2). Read this when a publish is refused,
when a weekly sweep files defects, or before re-pinning anything.

**No bundle ships unproven.** Every rule below fails closed: ambiguity is a
refusal, never a rubber stamp.

## The publish gate in one paragraph

`versta.batch` (the tool that writes the deployable catalog) refuses to
publish any bundle unless a pass-rated pack-acceptance scorecard
(`versta.pack-scorecard/1`) exists for the bundle's exact bytes
(`bundle.file` + `bundle.sha256`) and every direction in it clears, in
order: loads, the direction/chrF sanity floor, 100% digit preservation, and
the pinned golden-output band vs. the previously published bundle
(`pack-acceptance/bands.json`). The gate runs before any catalog write, so
a refusal leaves both `output/export/models.json` and the checked-in
`models.json` untouched.

## Thresholds (the gate's rules)

| Rule | Threshold | Rationale |
|---|---|---|
| Load | every direction `load_ok` | a pack that cannot load is dead on arrival |
| Direction sanity | corpus chrF ≥ per-direction `chrf_floor` | NEU-19 tripwire; a direction-mislabeled pack collapses through it (control: es-en≈22 on en-refs) |
| Digit preservation | `passed == total` (100%) | NEU-30 harm class: corrupted dates/prices/times are silent user damage |
| Golden band | chrF ≥ `golden_chrf − 3.0` (per direction, pinned) | cross-runner wiggle measured ≤0.6; real regressions move several points; mislabels collapse |
| Evidence identity | card joins on `bundle.file` + `bundle.sha256` | measurements bind to exact bytes; tampered bytes = no card = refusal |
| Number space | card `environment.corpus` and `metrics.chrf_signature` must equal the bands' pins | chrF is never comparable across corpora or metric parameterizations |
| Latency / RSS | measured and published, **not gating** | absolute budgets belong to the device-lab pass (NEU-63 phase 4) |

`3.0` is `chrf_max_drop` in `pack-acceptance/bands.json`; changing it changes
what may publish — treat it like moving a pin (dedicated commit).

## Operator flows

### Publish a new/updated bundle

```bash
cd translation

# 1. Build the bytes (dry run; gate skipped — the bundle is not evidence yet)
uv run python -m versta.batch --input_file models.json --output_dir ./output/export \
    --skip_publish_gate

# 2. Probe the produced bundle on the reference runner (arm64) and land cards:
#    this is what .github/workflows/pack-acceptance.yml does per commit; for a
#    manual publish, run the probe against the new tarball and commit the card:
uv run python -m versta.probe --bundle output/export/en-es-bundle.tar.gz \
    --probe_bin <leanmt-probe> --expect pass --output_dir output/probe
cp output/probe/scorecards/en-es.json scorecards/en-es.json   # re-pin commit, see below

# 3. Gated publish — refuses unless every bundle is proven
uv run python -m versta.batch --input_file models.json --output_dir ./output/export
```

`--skip_publish_gate` exists for development/dry-run exports only. Never use
it for artifacts meant for models.versta.app; its use is deliberate and
loud (stdout banner) for that reason.

Checks before publishing without the full batch: `uv run python -m
versta.batch.publish_gate --bundle <tar> [--scorecards_dir ...] [--bands_file ...]`;
exit 0 = publishable, 2 = refused with reasons.

### Weekly sweep (Monday 06:15 UTC, `.github/workflows/pack-sweep-weekly.yml`)

Fetches the pinned published packs (served bytes; a missing artifact is
itself a defect), re-packs + probes all of them on `ubuntu-24.04-arm`, runs
the adversarial negative control, evaluates every card against bands +
publish rules (`versta.probe.regress` → `output/sweep/defects.json`, schema
`versta.pack-defects/1`), uploads the evidence artifact, and files/dedupes
one GitHub issue per regressed slug (label `pack-acceptance`). Run it by
hand any time via Actions → pack-sweep-weekly → Run workflow.

The same-day device twin — the weekly pack-acceptance pass on the redroid
device lab (Monday 08:15 UTC), its canonical endpoints, and lab
etiquette — lives in `DEVICE-LAB-RUNBOOK.md`.

## Re-pin procedure (bands, scorecards, pins)

Same discipline the bench baselines follow (NEU-16/22/23). Pins move only
in dedicated commits; CI must never be "made green" by moving them.

1. Investigate first. A band/pin mismatch means upstream re-published
   underneath us, the corpus moved, the metric library moved, or the pack
   genuinely regressed. Name the cause before pinning anything.
2. Regenerate: `uv run python -m versta.probe.bands` (writes
   `pack-acceptance/bands.json` from `scorecards/`). Deterministic: rerun
   on a clean tree and diff — a moving diff without a moved scorecard means
   somebody hand-edited.
3. Land as a dedicated commit containing only pin/band data, with the
   measured scorecard diff in the commit message (e.g. `en-es: 81.02 →
   81.15`, evidence = workflow run link).
4. Record provenance on the commit: `git notes add -m "re-pin: <cause>;
   evidence: <run/artifact>; approver: <who>"` and push `refs/notes/*`.
5. `versta.probe.bands --check` runs in repo CI and in the test suite
   (`tests/test_bands.py::test_committed_bands_match_committed_scorecards`),
   so a band that no longer matches the committed scorecards fails the tree.

## Failure taxonomy

| Class | Signal | First action | Owner |
|---|---|---|---|
| Load failure | card `load_ok=false` | gate check refuses; get the leanmt error, reproduce locally with the same pinned probe binary | on-device pair (Tomasz); leanmt contract → Elin Sørensen |
| Direction mislabel (NEU-19) | `direction_check=fail`, chrF in the ~15–30 collapse band | treat as mirror of the adversarial control; inspect `base_model`/slug, re-export; never ship | on-device pair |
| Digit loss (NEU-30) | `digit_preservation` < 100%, `failing_ids` list | reproduce the listed corpus ids; check localization maps before touching the model | on-device pair (digit-map work: NEU-135) |
| Golden-band regression | chrF below `chrf_min` with floor intact | real model/engine decay or corpus drift; compare the two cards field by field | on-device pair, with pipeline pair when engine-side |
| Bundle/content drift | published-pin or catalog-md5 mismatch | registry/serving re-publish underneath us; escalate, **freeze publishes** until understood | on-device pair + Maya |
| Corpus drift | card corpus ≠ bands corpus | deliberate corpus bump forgot its re-baseline; re-pin from a full sweep | on-device + pipeline pair (Nikhil) |
| Metric-signature drift | `chrf_signature` mismatch | dependency bump changed chrF parameterization; deliberate re-baseline required | on-device + pipeline pair |
| Missing/unproven scorecard | gate refuses, "no passing scorecard" | the bundle was built but never probed; run the pack-acceptance workflow, land the card | whoever wants the publish |
| Harness defect | negative control passes, or probe crashes | the gate proves nothing; page Maya immediately, stop all publishes | Maya Lindqvist |
| Serving outage | fetch-report `missing` entries with kind `artifact-missing` (CDN answered 404-class) | CDN/registry problem; a pack that stops serving is paged as a defect | on-device pair |
| Transient CDN flake | fetch step exits 2, `cdn-fetch-flake` annotations + labeled step summary | infrastructure, not pack signal: re-run the job; only repeated flakes across runs are a serving outage | QA (Priya Nair) |

## Who gets paged

1. **Pack defects** (load/direction/digit/band/content drift): the weekly
   sweep files `pack-acceptance`-labeled issues; the on-device pair
   (Tomasz Nowak) owns triage within one business day.
2. **Scorecard/bands schema, eval-harness contracts**: pipeline pair
   (Nikhil Rao) — manifest-adjacent changes never land without pipeline-pair
   review (NEU-38).
3. **Harness integrity** (negative control untripped, probe binary drift,
   suspicious "all green"): page Maya Lindqvist directly; merges and
   closures are hers alone (NEU-126).
4. **Repeated weekly-sweep red (>2 consecutive runs) or publish freeze**:
   escalate to Big Boss with the defect census attached.

## Where the evidence lives

- `scorecards/`, `sweep-summary.json` — committed baseline cards (re-emitted
  by CI per run).
- `pack-acceptance/bands.json` — the pinned golden bands (this gate's data).
- `pack-acceptance/pins.json` — served-content pins (tar checksums) +
  leanmt commit the probe is built from.
- Workflow artifacts `pack-scorecards-<sha>` / `weekly-pack-sweep-<run>` —
  full sweep evidence, retained by GitHub (scorecards, defects.json,
  publish-gate verdict table).
