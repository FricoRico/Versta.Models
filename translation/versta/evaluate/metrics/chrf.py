import sacrebleu

from .base import Metric


class ChrfMetric(Metric):
    """chrF score implementation using sacrebleu."""

    def __init__(self):
        pass

    def name(self) -> str:
        return "chrf"

    def compute(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> float:
        """
        Compute chrF score using sacrebleu.

        Args:
            references: List of reference translations.
            hypotheses: List of generated translations.

        Returns:
            chrF score as a float.
        """
        if not references or not hypotheses:
            return 0.0

        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
        )
        return chrf.score
