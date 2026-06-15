from typing import TypedDict


class TonalTranslation(TypedDict):
    formal: str
    neutral: str
    casual: str


class ProcessedEntry(TypedDict):
    source: str
    target: str
    instruction: str
    input: str
    output: str
    method: str


class ExtractionResult(TypedDict):
    source: str
    target: str
    corpus: str
    num_pairs: int
    output_file: str


class CorpusConfig(TypedDict):
    corpus: str
    pairs: int | None
    release: str | None
    register: str | None


class LanguagePairConfig(TypedDict):
    source: str
    target: str
    synthetic: list[CorpusConfig]
    natural: list[CorpusConfig]
