# Pack acceptance scorecards (`versta.pack-scorecard/1`)

One JSON card per probed bundle, written by `versta.probe`
(`../versta/probe/scorecard.py`) and re-emitted on every pack-acceptance CI
run. Cards here are the committed first-sweep baselines; the CI workflow
overwrites them only via dedicated re-pin commits, so git history keeps the
full re-baseline trail even though the tree always holds the latest card.

Consumers (NEU-60 publish gate; the NEU-52 quality ledger when it wants
pack evidence) should dispatch on the top-level `schema` string and join
against eval results (`versta.eval-result/1`) on `bundle.sha256` — lowercase
hex on both sides. The aggregate `sweep-summary.json` (schema
`versta.pack-sweep/1`, written by `versta.probe.sweep`) pins the whole sweep:
per-bundle verdicts, chrF vs floor, digit preservation, latency and RSS in
one artifact — the epic thread's defect census is counted from it.

## Field guide

- `schema` / `scorecard_version` — `versta.pack-scorecard/1` and its integer
  version. Additive changes keep the string; shape breaks bump both.
- `bundle` — pack identity: file name, `sha256` (join key),
  `leanmt_binary_version`, `bundle_version`.
- `environment` — runner provenance (`host_arch`, `probe_arch`, `threads`,
  `runner`, `leanmt_commit`) plus corpus identity: `corpus` (name),
  `corpus_entries` (sentences scored), `corpus_sha256` (payload hash over
  the corpus files — see `versta/probe/corpus.py: corpus_payload_sha256`).
  chrF numbers are never comparable across corpora: the name and hash pin
  the exact corpus bytes.
- `metrics` — provenance of the chrF number space: `sacrebleu_version` and
  the verbatim `CHRF.get_signature()` of the installed reference library.
  The probe's chrF is a stdlib re-implementation parity-locked to that
  library (`tests/test_probe.py`); a dependency bump that changes the
  signature is a deliberate re-baseline, not a silent upgrade.
- `directions[]` — per direction: load verdict and timing, `chrf` vs the
  corpus-carried `chrf_floor` (`direction_check`), value-level digit
  preservation, p50/p95 latency, peak RSS, sentence/failure counts.
- `gates` — which measurements gate this card's `acceptance` (v1: `load` +
  `direction_chrf_floor`; digit preservation and latency/RSS are measured
  and published from the first sweep, and gate from NEU-60 on calibrated
  per-pair bands).
- `acceptance` / `expectation` / `expectation_met` — the v1 verdict, the
  run's declared expectation (`pass`, or `direction-fail` for the
  adversarial negative control), and whether reality matched it.
