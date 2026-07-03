import torch
from comet import download_model, load_from_checkpoint

from .base import Metric


class CometMetric(Metric):
    """COMET score using the official unbabel-comet library."""

    def __init__(self):
        model_path = download_model("Unbabel/wmt22-comet-da")
        self.model = load_from_checkpoint(model_path)

    def name(self) -> str:
        return "comet"

    def compute(
        self,
        references: list[str],
        hypotheses: list[str],
        sources: list[str] | None = None,
    ) -> float:
        data = [
            {"src": src or "", "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(
                sources or [""] * len(hypotheses), hypotheses, references
            )
        ]
        results = self.model.predict(
            data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0
        )
        return float(results.system_score)

    def sentence_scores(
        self,
        references: list[str],
        hypotheses: list[str],
        sources: list[str] | None = None,
    ) -> list[float]:
        data = [
            {"src": src or "", "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(
                sources or [""] * len(hypotheses), hypotheses, references
            )
        ]
        results = self.model.predict(
            data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0
        )
        return [float(s) for s in results.scores]
