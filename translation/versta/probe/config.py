"""
Fixed probe-run configuration.

Decode settings mirror the Versta app's translation defaults (see
``DEFAULT_NUMBER_OF_BEAMS``/``DEFAULT_MAX_SEQUENCE_LENGTH`` in
Versta.Android's TranslationPreferenceRepository): pack acceptance measures
the artifact as the app would run it — not under lab decode settings. The
translation cache is disabled so every probe sentence measures real
inference (same discipline as leanmt-bench).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeConfig:
    """Decode + measurement constants shared by bridge and scorecard."""

    beam: int = 4
    max_sequence_length: int = 256
    # Passed as leanmt-probe --threads; 0 keeps leanmt's process default.
    threads: int = 0


# The bundled probe corpus lives next to the package so the runner works from
# an editable install without extra flags.
DEFAULT_CORPUS_DIR = Path(__file__).parent / "corpora"

# Scorecard JSON schema version. Bump on incompatible shape changes; the
# schema is manifest-adjacent and reviewed with the pipeline pair.
SCORECARD_VERSION = 1

# Name the corpora use: bumped deliberately, like a bench baseline, when the
# probe set changes — chrF numbers from different corpus versions must never
# be silently mixed.
CORPUS_NAME = "traveler-v1"
