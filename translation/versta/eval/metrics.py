"""
Corpus metric scoring: sacreBLEU and chrF via sacrebleu.

sacrebleu is *the* reference implementation for these metrics (the values
Mozilla publishes for the same models are computed with it); the harness
records the library version and both metric signatures in every result, so
a score can be traced to the exact tokenization and parameterization that
produced it. Scores are corpus-level, single-reference, computed with
sacrebleu defaults (BLEU with the ``13a`` tokenizer; chrF2 with character
order 6 and beta 2) and rounded to 4 decimals — the precision the upstream
registry publishes.
"""

from dataclasses import dataclass

import sacrebleu

from sacrebleu.metrics import BLEU, CHRF

from typing import List


@dataclass(frozen=True)
class MetricScores:
    bleu: float
    chrf: float
    bleu_signature: str
    chrf_signature: str
    sacrebleu_version: str


def score_translations(hypotheses: List[str], references: List[str]) -> MetricScores:
    """
    Scores corpus translations against their pinned references.

    Args:
        hypotheses (list[str]): Engine translations, one per corpus line.
        references (list[str]): Reference translations, line-aligned with
            the hypotheses.

    Returns:
        MetricScores: BLEU and chrF corpus scores plus the sacrebleu
        version and metric signatures that produced them.

    Raises:
        ValueError: If the inputs are empty or not line-aligned.
    """
    if not hypotheses or len(hypotheses) != len(references):
        raise ValueError(
            f"scoring requires a non-empty, line-aligned hypothesis/"
            f"reference pair (got {len(hypotheses)} vs {len(references)})"
        )

    bleu = BLEU()
    chrf = CHRF()

    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    return MetricScores(
        bleu=round(bleu_score.score, 4),
        chrf=round(chrf_score.score, 4),
        bleu_signature=str(bleu.get_signature()),
        chrf_signature=str(chrf.get_signature()),
        sacrebleu_version=sacrebleu.__version__,
    )
