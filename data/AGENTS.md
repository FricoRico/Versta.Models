# data/

Shared runtime data bundles for the Versta app. Read the root `/AGENTS.md`
first; it defines the shared Python, typing and CLI conventions this module
follows.

## Module purpose

This module packages non-model runtime data into the tarballs the app
side-loads alongside its models. The current bundle is `versta-tts-data`:

- **espeak-ng data** — built from source (`git@github.com:espeak-ng/espeak-ng.git`) via CMake's `data` target, producing `espeak-ng-data`.
- **open-jtalk dictionary** — downloaded as a tarball (`open_jtalk_dic_utf_8-1.11.tar.gz` from SourceForge) and extracted.

The espeak build shells out to `git` and `cmake`, so both must be installed on
the host. Dependencies otherwise stay minimal: `setuptools`, `requests`.

## Tools

Run from within this directory with `requirements.txt` installed.

### `versta.tts`

Downloads espeak-ng, builds its data via CMake, downloads the open-jtalk
dictionary and generates bundle metadata:

```bash
python -m versta.tts --output_dir ./output --temp_dir ./tmp
```

- `--output_dir`: defaults to `output/`; the bundle staging root is `output/versta-tts-data`.
- `--temp_dir`: defaults to `tmp/`; used for git clones and tarball downloads (gitignored).
- `--keep_intermediates`: keep the CMake build directory.
- `--keep_downloads`: keep the cloned repos/downloaded tarballs.
- `tts/download.py` provides `download_folder_from_git` (clone via SSH and extract a folder) and `download_folder_from_tarball`; `tts/espeak.py:build_data` runs the CMake configure/build and moves `espeak-ng-data` into place.

### `versta.bundle`

Bundles the staged data directory into a tarball plus `.sha256` checksum,
using the same bundle package shape as the other modules
(`bundle/bundle_tar.py`, `bundle/metadata.py`).

- The checked-in `data.json` is the app-facing catalog for this bundle (id `versta-tts-data`, type `tts`), pointing at `models.versta.app/data/...`. Regenerate it when the bundle contents change.

## Conventions specific to this module

- `version.txt` lives at `versta/version.txt` (currently `v1.0.0`), read at import time by each `__main__.py`. Bump it when the bundle contents or metadata schema change.
- The espeak-ng build requires a native toolchain on the host (CMake, compiler). Do not vendor build outputs into git; they are generated artifacts.
- The SSH clone of espeak-ng implies host git credentials; do not switch to unauthenticated mirrors without checking the security implications against the repo's privacy-first policy.
- Strict typing holds here as everywhere: annotate every function; structured metadata goes through `metadata.py`/TypedDict shapes, not loosely typed dicts.

## Verification

No test suite. Smoke-check by running the full pipeline into the gitignored
trees and confirming `espeak-ng-data`, `open-jtalk-data` and the tarball:

```bash
python -m versta.tts --output_dir output --temp_dir tmp
```
