"""
Hash-pinned evaluation reference corpora (see ``evals/corpora/README.md``).

Producer side: ``build_snapshot`` / ``fetch_snapshot`` / ``verify_corpus``
create and reproduce the pinned snapshot byte-identically, failing closed on
any hash or count drift. Consumer side: ``resolve_slice`` hands the eval
harness one direction's re-verified source and reference sentences.
"""

from .pipeline import build_snapshot, fetch_snapshot, verify_corpus
from .slices import (
    CorpusError,
    CorpusSlice,
    load_manifest,
    resolve_slice,
    snapshot_dir,
)

__all__ = [
    "CorpusError",
    "CorpusSlice",
    "build_snapshot",
    "fetch_snapshot",
    "load_manifest",
    "resolve_slice",
    "snapshot_dir",
    "verify_corpus",
]
