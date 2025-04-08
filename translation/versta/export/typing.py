from pathlib import Path
from typing import TypedDict

class ORTFiles(TypedDict):
    encoder: Path
    decoder: Path

class TokenizerFiles(TypedDict):
    config: Path
    vocabulary: Path
    source: Path
    target: Path