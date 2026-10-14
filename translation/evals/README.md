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
