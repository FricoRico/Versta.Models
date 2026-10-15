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

`revision` is recorded in every eval result; bumping the corpus pin is a
deliberate event that makes old/new scores non-comparable by construction.

Consumer rule: unknown fields may appear as the schema evolves additively;
consumers must ignore them rather than fail.

Additive extension (NEU-52): `model.size_bytes` — bytes of the artifact
under evaluation (the tarball for bundle runs; the sum of
model/vocabulary/shortlist payloads for direction-dir runs). Recorded so
the ledger and the generated eval summary carry size notes without
touching the filesystem.

## Direction ledger (`evals/ledger/<direction>.json`, schema `versta.eval-ledger/1`)

The checked-in, append-only history of shipped versions for one direction —
the model-registry history the pipeline charter requires, and the
predecessor source for every release gate run. Written exclusively by
`versta.eval.gate --record`; never hand-edited (a mistaken record is
corrected by a new entry, not by rewriting history).

```json
{
  "schema": "versta.eval-ledger/1",
  "direction": "en-es",
  "source_language": "en",
  "target_language": "es",
  "entries": [
    {
      "version": "v2.0.0",
      "result": { "...versta.eval-result/1 document..." : "..." },
      "gate": {
        "decision": "pass",
        "threshold_chrf": 1.0,
        "predecessor_version": "v1.12.14",
        "delta_bleu": 0.31,
        "delta_chrf": 0.87,
        "delta_comet22": null,
        "reason": null,
        "decided_utc": "2026-10-14T12:00:00Z"
      }
    }
  ]
}
```

Rules enforced fail-fast on load — a corrupt ledger is a stop-the-line
event, because the gate must never compare against history it cannot
trust:

- `decision` is one of `baseline` | `pass` | `override` | `rebaseline`;
  `override` and `rebaseline` entries **must** carry a non-empty written
  `reason` (machine-checked);
- each embedded `result` is a full `versta.eval-result/1` document — the
  bundle checksum, corpus pin, engine version and size notes live there;
- versions are never re-recorded: appending an existing version is an
  error, and a gate re-run for an already-recorded version compares
  against the previous entry, never against itself.

## The release gate (`uv run python -m versta.eval.gate`)

```
uv run python -m versta.eval.gate --result output/evals/en-es.json \
    --ledger_dir evals/ledger [--threshold 1.0] \
    [--reason "..."] [--rebaseline] [--record] [--summary_out PATH]
```

Hermetic by construction — the gate reads only the candidate result JSON
and the direction ledger, so it runs identically on a laptop and in CI.
Decisions:

| Situation | Outcome |
|---|---|
| No recorded predecessor | `baseline` (ships) |
| chrF delta ≥ −threshold (default 1.0, `versta/eval/config.py`) | `pass` (ships) |
| chrF delta < −threshold | `fail`, exit 1 — unless `--reason` given → `override` (ships) |
| Corpus pin changed vs. predecessor | `fail`, exit 1 — scores across pins are not comparable; `--rebaseline --reason` → `rebaseline` (ships) |
| `--reason` on a clean pass/baseline | usage error, exit 2 — a reason always justifies something concrete |

`--record` appends a shipping decision (`baseline`/`pass`/`override`/
`rebaseline`) to the checked-in ledger; a failed gate is never recorded.
Every run writes the **generated release eval-summary markdown** (default
`output/evals/<direction>-<version>-eval-summary.md`, exit-code path
included): decision banner, metric table with deltas vs. the predecessor,
corpus pin, engine version, artifact checksum and size notes, and the
recorded reason for override/rebaseline decisions. Posting that file with
the release satisfies the charter's "every model release posts an eval
summary" without handwritten prose drifting from the numbers.

## CI wiring (NEU-53)

Two jobs, deliberately split:

- **PR CI stays hermetic** (`.github/workflows/ci.yml`, the required
  `test` check): the whole pytest suite — gate decisions, metrics,
  ledger, corpus manifest validation — runs against the tiny synthetic
  fixture pair and a fake engine, in seconds, with no network. A
  `conftest.py` guard fails any test that attempts a real connection, so
  the hermetic property is enforced, not aspirational; branch protection
  (NEU-34) stays green and cheap.
- **Release candidates run the real thing**
  (`.github/workflows/release-eval.yml`, `workflow_dispatch` on the
  release-candidate branch): fetches the hash-pinned corpus snapshot,
  downloads and SHA-256-verifies the candidate bundle, translates through
  the digest-pinned `bergamot-translator` image with the fixed decode
  settings, runs the gate and — on a shipping decision — appends the
  ledger entry and pushes it back to the release branch. Wall-clock per
  stage goes to the run summary alongside the generated eval-summary
  markdown; the gate's exit code is the job result, so **a failing gate
  blocks the release candidate**.

Dispatch inputs (`source`, `target`, `bundle_url` + `bundle_sha256`,
`bergamot_image`, optional `comet`/`threshold`/`reason`/`rebaseline`/
`record`) mirror the CLI flags above. Reproduce a CI run locally with the
same three commands shown in the job log (`versta.eval.corpora fetch` →
`versta.eval` → `versta.eval.gate`); everything the numbers depend on is
pinned in-repo or recorded in the result JSON, so the local run matches.
