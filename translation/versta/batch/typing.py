from pathlib import Path
from typing import TypedDict


class ModelFile(TypedDict):
    source_language: str
    target_language: str
    architecture: str
    score: float
    version: str
    size: int
    bundle: str
    checksum: str


class ExportedModel(TypedDict):
    path: Path
    source_language: str
    target_language: str
    architecture: str
    score: float
    version: str


class ExportedBundle(TypedDict):
    path: Path
    checksum: Path
    source_language: str
    target_language: str
    architecture: str
    bidirectional: bool
    score: float
    version: str
