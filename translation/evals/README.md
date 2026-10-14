# evals/ — quality evaluation contracts and history

Quality gates for the shipped translation models (project spec: NEU-44).
This directory carries the checked-in contracts every other piece builds on:

- `corpora/flores200-dev.json` — the pinned FLORES-200 reference snapshot
  manifest (provenance, artifact mirrors, per-file hashes and sentence
  counts per direction). Produced and bumped by the corpus pinning tooling
  (`uv run python -m versta.eval.corpora`, docs in `corpora/README.md`).
- Eval result JSON — one per evaluated direction, written by
  `uv run python -m versta.eval` (schema `versta.eval-result/1`, below);
  consumed by the release gate and the ledger (NEU-52).
- `ledger/` — the append-only quality history the gate compares release
  candidates against (landed by NEU-52/NEU-53).

All three are machine-checked contracts; extend additively, never rename or
drop fields, and bump the `schema` string on any breaking change.

## Corpus snapshot (`corpora/flores200-dev.json`, schema `versta/corpus-manifest@1`)

The manifest pins a repackaged snapshot of the upstream FLORES-200 `dev`
split covering every shipped direction. It records:

- `artifact` — the snapshot tarball's file name, SHA-256, size and fetch
  mirrors (canonical bucket first, GitHub release mirror second);
- `files` — every payload file inside the snapshot with its SHA-256 and
  sentence count;
- `directions` — per shipped direction (`en-es`, `es-en`, `mt-lv`, …): the
  payload files for source and reference plus the aligned sentence count.

Payload bytes are not in git. Fetch them (hash-verified before extraction,
re-verified after) into the gitignored cache:

```bash
uv run python -m versta.eval.corpora fetch     # → output/corpora/flores200-dev-r1/
uv run python -m versta.eval.corpora fetch --check   # verify the cached copy only
```

Full provenance, license and the revision-bump procedure:
[`corpora/README.md`](corpora/README.md).

The harness (`versta.eval`) loads this manifest and resolves one direction's
slice via `versta.eval.corpora.resolve_slice`, re-verifying the payload
hashes at scoring time. Rules enforced before any engine time is spent:

- the direction must have a manifest entry — a direction added to
  `models.json` without corpus coverage fails the eval explicitly;
- every payload file's SHA-256 must match the pin or the run aborts —
  evals against an unpinned or stale corpus are never scored;
- sentence counts must match the manifest exactly, and source lines must
  be non-blank (engine output is scored line-aligned with the source).

## Eval result (`versta.eval-result/1`)

One JSON document per evaluation, written atomically (sorted keys) to
`output/evals/<direction>.json`:

```json
{
  "schema": "versta.eval-result/1",
  "direction": "en-es",
  "source_language": "en",
  "target_language": "es",
  "model": {
    "base_model": "Firefox/en-es",
    "version": "v2.0.0",
    "architecture": "tiny",
    "model_sha256": "…",
    "metadata_sha256": "…",
    "bundle_file": "en-es-bundle.tar.gz",
    "bundle_sha256": "…"
  },
  "corpus": {
    "corpus": "flores200-dev",
    "split": "dev",
    "revision": "r1",
    "sentences": 997,
    "source_sha256": "…",
    "target_sha256": "…"
  },
  "engine": {
    "name": "bergamot-translator",
    "variant": "docker",
    "version": "…",
    "image": "…",
    "decode": { "beam-size": 1, "cpu-threads": 4, "…": "…" },
    "config_sha256": "…"
  },
  "metrics": {
    "sacrebleu_version": "…",
    "bleu_signature": "nrefs:…|bs:…",
    "chrf_signature": "nrefs:…|bs:…",
    "comet22_model": null
  },
  "scores": { "bleu": 41.2, "chrf": 65.3, "comet22": null },
  "run": {
    "started_utc": "2026-10-14T00:00:00Z",
    "duration_seconds": 812.4,
    "hypothesis_sha256": "…"
  }
}
```

Reading it:

- Reproducibility contract: same artifact + same corpus pin + same engine
  and decode config ⇒ identical `scores` and `run.hypothesis_sha256` on
  re-run; `run.duration_seconds`/`started_utc` are wall-clock only.
- `corpus.revision` + payload hashes make scores comparable across time;
  a result pinned to a different corpus revision is not comparable to an
  `r1` result (the gate fails closed on that).
- `metrics.*_signature` capture the exact sacreBLEU tokenization/options,
  so results stay comparable even if the library defaults ever shift.
- `scores.comet22` is `null`, never `0.0`, when the optional COMET-22 tier
  was not run — absence is explicit.

Consumer rule: unknown fields may appear as the schema evolves additively;
consumers must ignore them rather than fail.
