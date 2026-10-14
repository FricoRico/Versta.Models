"""
Reproducible quality evaluation for produced translation models.

The harness translates the pinned FLORES-200 reference snapshot (see
``evals/corpora/flores200-dev.json`` and the ``versta.eval.corpora``
package that fetches and verifies it) through the reference runtime —
the upstream bergamot-translator CLI until the LeanMT batch translate
mode (NEU-50) lands — and scores the output with sacreBLEU and chrF,
emitting a versioned JSON result the release gate and the ``evals/``
ledger consume (see NEU-44 proposal). Scores measure real translations
from the shipped artifact, so output-affecting defects (NEU-30 class)
surface here where packaging checks cannot see them.
"""
