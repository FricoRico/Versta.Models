from typing import TypedDict

class ORTFiles(TypedDict):
    encoder: str
    decoder: str

class TokenizerFiles(TypedDict):
    config: str
    vocabulary: str
    source: str
    target: str