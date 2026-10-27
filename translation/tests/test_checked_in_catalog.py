import json
from pathlib import Path
from typing import Any

from versta.schema import validate_catalog

CATALOG_PATH = Path(__file__).resolve().parent.parent / "models.json"


def _load_checked_in_catalog() -> list[Any]:
    return json.loads(CATALOG_PATH.read_text())


class TestCheckedInCatalog:
    def test_validates_against_app_loader_contract(self) -> None:
        validate_catalog(_load_checked_in_catalog())

    def test_base_model_matches_declared_direction(self) -> None:
        for group in _load_checked_in_catalog():
            for entry in group:
                expected = (
                    f"Firefox/{entry['source_language']}-{entry['target_language']}"
                )
                assert entry["base_model"] == expected, (
                    f"direction {entry['source_language']}-{entry['target_language']}: "
                    f"base_model {entry['base_model']!r} should be {expected!r}"
                )
