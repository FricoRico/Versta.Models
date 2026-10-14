"""
Fixed evaluation-time configuration.

The decode settings mirror what Firefox ships for these models in
production (the bergamot config derived from
``extension/controller/translation/translationWorker.js`` and used by
Mozilla's own model evaluations): greedy decoding (``beam-size: 1``),
sentence-level input splitting and the int8 GEMM precision selected by
model filename. Evals must never silently drift from these values — they
are package constants, not CLI flags, and every result JSON records them
under ``engine.decode``.

``cpu_threads`` is the one deliberate deviation: production passes ``0``
(auto-detect). The reference runtime is pinned to a fixed thread count so
batched decoding is scheduled identically on every machine; the int8
alphas models this harness evaluates compute integer GEMMs whose results
do not depend on thread count, but pinning costs nothing and removes a
whole class of cross-machine doubt.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DecodeConfig:
    bergamot_mode: str = "wasm"
    beam_size: int = 1
    normalize: float = 1.0
    word_penalty: float = 0.0
    max_length_break: int = 128
    mini_batch_words: int = 1024
    workspace: int = 128
    max_length_factor: float = 2.0
    skip_cost: bool = True
    cpu_threads: int = 4
    alignment: str = "soft"
    ssplit_mode: str = "sentence"
    max_input_sentence_tokens: int = 1024
    max_input_tokens: int = 2048000


DEFAULT_DECODE_CONFIG = DecodeConfig()

# Model archive name suffix -> bergamot gemm-precision, mirroring the
# selection rule in Mozilla's eval harness (translators/bergamot.sh):
# models trained with intgemm alphas need the alpha-shifted int8 paths.
_GEMM_PRECISION_BY_SUFFIX: Dict[str, str] = {
    ".intgemm.alphas.bin": "int8shiftAlphaAll",
    ".intgemm8.bin": "int8shiftAll",
}


class EvalConfigError(ValueError):
    """A produced model or corpus cannot be mapped onto the fixed eval config."""


def resolve_gemm_precision(model_file_name: str) -> str:
    """
    Resolves the bergamot ``gemm-precision`` for a quantized model file name.

    Args:
        model_file_name (str): Bare file name of the binary model, e.g.
            ``model.enes.intgemm.alphas.bin``.

    Returns:
        str: The bergamot gemm-precision identifier.

    Raises:
        EvalConfigError: If the file name matches no known quantized layout
            (an unrecognized model build — fail closed rather than guessing
            a precision that would silently corrupt scores).
    """
    for suffix, precision in _GEMM_PRECISION_BY_SUFFIX.items():
        if model_file_name.endswith(suffix):
            return precision
    raise EvalConfigError(
        f"cannot resolve gemm-precision for model file '{model_file_name}': "
        f"expected a name ending in one of {sorted(_GEMM_PRECISION_BY_SUFFIX)}"
    )
