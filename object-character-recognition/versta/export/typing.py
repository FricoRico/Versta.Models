from typing import Dict, List, Literal, TypedDict

ModelKind = Literal["detector", "recognizer", "scriptClassifier", "textlineOrientation"]

ManifestRole = Literal[
    "detector", "recognizer", "scriptClassifier", "textlineOrientation", "keys"
]


class ModelSpec(TypedDict):
    stem: str
    url: str
    kind: ModelKind
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
