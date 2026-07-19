from pathlib import Path
from typing import TypedDict

class LanguageModelFilesMetadata(TypedDict):
    model: str
    vocabulary: str
    target_vocabulary: str | None
    shortlist: str

class BundleMetadata(TypedDict):
    directory: Path
    source_language: str
    target_language: str
    architecture: str
    base_model: str
    score: float
    version: str
    files: LanguageModelFilesMetadata
