import json
from pathlib import Path
from typing import List, TypedDict

MODELS_JSON = Path(__file__).parent.parent.parent / "models.json"
STORAGE_BASE_URL = "https://models.versta.app/object-character-recognition"


class CatalogEntry(TypedDict):
    id: str
    name: str
    base_model: str
    architectures: List[str]
    size: int
    version: str
    bundle: str
    checksum: str
    languages: List[str]


def update_catalog(
    unique_id: str, version: str, bundle_file: Path, checksum_file: Path
) -> Path:
    """
    Updates the checked-in models.json catalog entry for the OCR pack with
    the produced bundle's version, size and object-storage URLs. Fields that
    describe the model itself (id, name, architectures, languages) are
    preserved untouched.

    Args:
        unique_id (str): The catalog entry id to update.
        version (str): The bundle version (from versta/version.txt).
        bundle_file (Path): The produced bundle tarball.
        checksum_file (Path): The produced checksum file.

    Returns:
        Path: The written models.json path.

    Raises:
        ValueError: If the catalog contains no entry with the given id.
    """
    with open(MODELS_JSON, "r") as f:
        entries: List[CatalogEntry] = json.load(f)

    matches = [entry for entry in entries if entry["id"] == unique_id]
    if not matches:
        raise ValueError(f"No models.json entry with id '{unique_id}'")
    entry = matches[0]

    entry["base_model"] = "PaddlePaddle/PP-OCRv6"
    entry["size"] = bundle_file.stat().st_size
    entry["version"] = version
    entry["bundle"] = f"{STORAGE_BASE_URL}/{version}/{bundle_file.name}"
    entry["checksum"] = f"{STORAGE_BASE_URL}/{version}/{checksum_file.name}"

    with open(MODELS_JSON, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return MODELS_JSON
