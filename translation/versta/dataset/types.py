from typing import TypedDict


class TonalTranslation(TypedDict):
    formal: str
    informal: str
    casual: str


class ProcessedEntry(TypedDict):
    source: str
    target: str
    instruction: str
    input: str
    output: str


class ExtractionResult(TypedDict):
    source: str
    target: str
    corpus: str
    num_pairs: int
    output_file: str


class MultiCorpusConfig(TypedDict):
    corpus: str
    pairs: int | None
    release: str | None
