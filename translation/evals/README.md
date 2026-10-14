# evals/ — quality evaluation contracts and ledger

Quality gates for the shipped translation models (project spec: NEU-44).
This directory carries the two artifacts every other piece builds on:

- `corpora.json` + `corpora/` — the pinned FLORES-200 reference snapshot
  (provenance, hashes, sentence counts per direction). Produced by the
  corpus pinning task (NEU-49).
- Result JSON files — one per evaluated direction, written by
  `versta.eval` (NEU-51) and collected into the release ledger by the
  gate task (NEU-52).

Both are machine-checked contracts; extend additively, never rename or
drop fields, bump the `schema` string on any breaking change.

## Corpus manifest (`corpora.json`, schema `versta.eval-corpora/1`)

```json
{
  "schema": "versta.eval-corpora/1",
  "corpus": "flores-200",
  "revision": "<upstream pin, e.g. flores commit hash>",
  "split": "dev",
  "license": "CC-BY-SA-4.0",
  "directions": {
    "en-es": {
      "flores_source": "eng_Latn",
      "flores_target": "spa_Latn",
      "sentences": 997,
      "source": { "path": "flores-200@<rev>/dev/eng_Latn.dev", "sha256": "<hex>" },
      "target": { "path": "flores-200@<rev>/dev/spa_Latn.dev", "sha256": "<hex>" }
    }
  }
}
```

Rules the harness enforces before scoring:

- payload paths are relative to `evals/corpora/` (never absolute);
- every payload file's SHA-256 must match the pin or the run aborts —
  scoring against an unverified corpus is worse than no score;
- line counts must match `sentences`, and source files carry no blank
  lines (engine output must stay line-aligned with the input).

`revision` is recorded in every eval result; bumping the corpus pin is a
deliberate event that makes old/new scores non-comparable by construction.

## Result JSON (schema `versta.eval-result/1`)

Written by `uv run python -m versta.eval --source en --target es --bundle
…` (default destination `output/evals/<src>-<tgt>.json`). Fields:

| Field | Meaning |
|---|---|
| `schema` | `versta.eval-result/1` |
| `direction`, `source_language`, `target_language` | The evaluated direction |
| `model` | `base_model`, `version`, `architecture`, `model_sha256`, `metadata_sha256`; plus `bundle_file`/`bundle_sha256` when evaluated from a shipped bundle |
| `corpus` | `corpus`, `split`, `revision`, `sentences`, `source_sha256`, `target_sha256` |
| `engine` | `name`, `variant` (`local`/`docker`), `version`, `decode` (fixed decode settings), `config_sha256`; `image` for container runs |
| `metrics` | `sacrebleu_version`, `bleu_signature`, `chrf_signature`, `comet22_model` (null unless scored) |
| `scores` | `bleu`, `chrf` (4-decimal corpus scores, sacrebleu defaults) and `comet22` (null unless `--comet`) |
| `run` | `started_utc`, `duration_seconds`, `hypothesis_sha256` (hash of the scored translation file) |

Reproducibility contract: same artifact + same corpus pin + same engine
and decode config ⇒ identical `scores` and `hypothesis_sha256` on re-run.
The gate (NEU-52) compares `scores` against the predecessor's ledger
entry; everything else in the document is audit trail for *why* numbers
moved when they do.

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
