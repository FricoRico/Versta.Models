from typing import TypedDict


class ScraperEntry(TypedDict):
    source: str
    target: str
    input: str
    output: str
    category: str


class ScrapeIntermediate(TypedDict):
    text: str
    language: str
    category: str
    title: str
    sentence_index: int
