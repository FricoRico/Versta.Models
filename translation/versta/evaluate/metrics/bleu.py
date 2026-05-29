import sacrebleu

from .base import Metric


class BleuMetric(Metric):
    """BLEU score implementation using sacrebleu."""

    def __init__(self):
        pass

    def name(self) -> str:
        return "bleu"

    def compute(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> float:
        """
        Compute BLEU score using sacrebleu.

        Args:
            references: List of reference translations.
            hypotheses: List of generated translations.

        Returns:
            BLEU score as a float.
        """
        if not references or not hypotheses:
            return 0.0

        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
        )
        return bleu.score
