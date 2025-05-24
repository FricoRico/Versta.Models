from pathlib import Path
from typing import TypedDict, List

class ModelFile(TypedDict):
    base_model: str
    source_language: str
    target_language: str
    bidirectional: bool
    architectures: List[str]
    score: float
    version: str
    size: int
    bundle: str
    checksum: str

class ExportedModel(TypedDict):
    path: Path
    base_model: str
    source_language: str
    target_language: str
    architectures: List[str]
    score: float
    version: str

class ExportedBundle(TypedDict):
    path: Path
    checksum: Path
    base_model: str
    bidirectional: bool
    source_language: str
    target_language: str
    architectures: List[str]
    score: float
    version: str