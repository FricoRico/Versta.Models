from pathlib import Path
from typing import NotRequired, TypedDict


class GenerationConfig(TypedDict):
    temperature: float
    top_p: float
    min_p: float
    top_k: int
    do_sample: bool
    repetition_penalty: float
    no_repeat_ngram_size: int


class EvaluationConfig(TypedDict):
    model: str
    dataset: str
    output: Path
    source: str
    target: str
    percentage: float
    batch_size: int
    max_seq_length: int
    tones: list[str]
    use_comet: bool
    use_bleu: bool
    use_chrf: bool
    gen_config: GenerationConfig
    model_type: NotRequired[str]
    device: NotRequired[str | None]


class ToneResults(TypedDict):
    comet: float | None
    bleu: float | None
    chrf: float | None


class SentenceScore(TypedDict):
    source: str
    reference: str
    hypothesis: str
    tone: str | None
    bleu: float | None
    chrf: float | None
    comet: float | None


class EvaluationResult(TypedDict):
    config: EvaluationConfig
    overall: ToneResults
    by_tone: dict[str, ToneResults]
    sentence_scores: list[SentenceScore]
    num_samples: int
    model_name: str
    dataset_type: str
    dataset_name: str
    timestamp: str
