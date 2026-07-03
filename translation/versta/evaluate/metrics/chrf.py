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
        if not references or not hypotheses:
            return 0.0

        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
        )
        return chrf.score

    def sentence_scores(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> list[float]:
        if not references or not hypotheses:
            return []
        return [
            sacrebleu.sentence_chrf(hyp, [ref]).score
            for hyp, ref in zip(hypotheses, references)
        ]
