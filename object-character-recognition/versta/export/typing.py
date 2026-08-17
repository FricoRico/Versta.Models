from typing import NotRequired, Dict, List, Literal, TypedDict

ModelKind = Literal[
    "detector",
    "recognizer",
    "scriptClassifier",
    "textlineOrientation",
    "aligner",
    "glyphmatte",
]

ManifestRole = Literal[
    "detector",
    "recognizer",
    "scriptClassifier",
    "textlineOrientation",
    "keys",
    "aligner",
    "glyphmatte",
]


class HfSource(TypedDict):
    repo_id: str
    filename: str


class ModelSpec(TypedDict):
    stem: str
    url: NotRequired[str]
    hf: NotRequired[HfSource]
    kind: ModelKind
    note: NotRequired[str]
    opset: int
    pir: bool
    tier: str
    script: str


class ManifestFile(TypedDict, total=False):
    name: str
    sizeBytes: int
    sha256: str
    role: ManifestRole
    script: str
    priority: int
    note: str


class Manifest(TypedDict):
    version: str
    pack: str
    files: List[ManifestFile]


FoldVariants = Dict[str, List[str]]
