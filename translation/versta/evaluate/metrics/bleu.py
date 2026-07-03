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
        if not references or not hypotheses:
            return 0.0

        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
        )
        return bleu.score

    def sentence_scores(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> list[float]:
        if not references or not hypotheses:
            return []
        return [
            sacrebleu.sentence_bleu(hyp, [ref]).score
            for hyp, ref in zip(hypotheses, references)
        ]
