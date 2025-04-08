from pathlib import Path
from typing import TypedDict, List

class BundleMetadata(TypedDict):
    directory: Path
    source_language: str
    target_language: str

class LanguageTokenizerFiles(TypedDict):
    config: Path
    vocabulary: Path
    source: Path
    target: Path

class LanguageORTFiles(TypedDict):
    encoder: Path
    decoder: Path

class LanguageFiles(TypedDict):
    tokenizer: LanguageTokenizerFiles
    inference: LanguageORTFiles

class LanguageMetadata(TypedDict):
    source_language: str
    target_language: str
    architectures: List[str]
    files: LanguageFiles