# Evaluation corpora

Hash-pinned snapshots of the reference corpora the per-pair quality gates
score against (`versta.eval`, project on NEU-44). The manifest in this
directory is the contract: **same manifest in, byte-identical corpus out** —
on a developer machine, in CI, or in a release job.

## What is pinned

| | |
|---|---|
| Dataset | FLORES-200, `dev` split (aligned, professionally translated; 997 sentences per language) |
| Coverage | the 63 shipped directions of `translation/models.json`: 31 bidirectional pairs plus the single `Firefox/mt-lv` direction — 34 languages |
| Snapshot | `flores200-dev-r1.tar.gz` (sha256 `ab517371931a26060b21e01eb5ae8e3d8d164fdb518a09fdd50944ed829cb9e7`, ~2 MB) |
| Manifest | [`flores200-dev.json`](flores200-dev.json) — revision, upstream provenance, artifact mirrors, per-file hashes + sentence counts, per-direction entries |

`manifest.json` inside the artifact mirrors the same per-file pins, so an
artifact stays self-describing even when copied out of band.

## Provenance & license

Built from the upstream FLORES-200 release tarball
`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`
(sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`,
2022-07-14). Language files are redistributed **unmodified** (exact upstream
bytes); the artifact only repacks the 34 needed languages and adds a `NOTICE`.

FLORES-200 is licensed **CC-BY-SA 4.0** (Meta AI / NLLB team). The attribution
string ships in the artifact `NOTICE` and in the manifest's
`upstream.attribution`. Keep attribution intact when re-hosting.

## Layout of the snapshot

```
flores200-dev-r1/
├── NOTICE                  # provenance + CC-BY-SA 4.0 attribution
├── manifest.json           # embedded pin: revision + per-file hashes/counts
└── dev/<flores_code>.dev   # one file per language, e.g. dev/eng_Latn.dev
```

`flores200-dev.json` maps catalog language codes to FLORES-200 codes
(`language_map`: `en` → `eng_Latn`, `fa` → `pes_Arab`, `zh-Hans` → `zho_Hans`,
…). `directions[]` records for each shipped direction which file is the
source and which the reference, plus the aligned sentence count.

### The mt-lv entry

`models.json` ships one group whose fields are inconsistent with its bundled
model (`base_model: Firefox/mt-lv`, fields `lv`→`en`); reconciling the catalog
is NEU-19 scope. For evaluation the corpus follows the **bundled model**:
direction `mt`→`lv`, files `mlt_Latn.dev` / `lvs_Latn.dev`. `en`/`lv` are
already covered by the pair side.

## Using the corpus

```bash
cd translation
uv sync

uv run python -m versta.eval.corpora fetch     # download + verify (sha256-pinned)
uv run python -m versta.eval.corpora fetch --check   # verify the cached copy only
```

The corpus lands in the gitignored `output/corpora/flores200-dev-r1/`. Fetches
try the mirrors recorded in the manifest in order
(`models.versta.app/evals/corpora/…` canonical, GitHub release mirror second)
and verify the artifact hash **before** extracting; every extracted file is
re-hashed and line-counted against the manifest. Any mismatch aborts — a
corrupt or stale corpus never silently feeds an eval.

## Updating (new revision)

1. Bump `--revision` (r2, …). Never mutate a published revision — eval results
   cite revision + hashes and must stay comparable across time.
2. Build from the pinned upstream tarball:

   ```bash
   uv run python -m versta.eval.corpora build \
       --upstream_tarball /path/to/flores200_dataset.tar.gz \
       --revision 2
   ```

   The build fails fast if a needed language is missing or a dev file does not
   carry exactly 997 sentences. Rebuilding from the same inputs reproduces the
   same artifact hash (deterministic packing; `SOURCE_DATE_EPOCH` pins the
   archive timestamp).
3. Publish the artifact: upload `output/corpora/flores200-dev-r2.tar.gz` to
   `models.versta.app/evals/corpora/` (object-storage access required — same
   manual deploy step as model bundles) and attach both the artifact and its
   `.sha256` to a GitHub release tagged `evals-corpora-flores200-dev-r2` as
   the access-free mirror.
4. Commit the rewritten `evals/corpora/flores200-dev.json` together with this
   README in the same PR.

Changing *which* upstream dataset to snapshot (e.g. moving to the revised
FLORES+ corpus, `openlanguagedata/flores_plus`) is a format-level decision:
new revision, updated `upstream` block, and a note here — scores pinned to
different corpora are not comparable.
