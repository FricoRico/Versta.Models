from pathlib import Path
from typing import Optional, TypedDict


class ModelFile(TypedDict):
    """One requested translation direction in the batch input catalog."""

    source_language: str
    target_language: str
    architecture: Optional[str]
    score: float


class CatalogEntry(TypedDict):
    """One entry in the generated (pre-sync) catalog written by save_model_file."""

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
