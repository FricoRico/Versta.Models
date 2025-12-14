from typing import TypedDict, List

class OCRBundleMetadata(TypedDict):
    directory: str
    languages: List[str]
    module: str
