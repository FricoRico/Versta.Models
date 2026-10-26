from typing import List

from ..language_codes import bundle_languages, normalize_language
from .metadata import BundleMetadata


def assert_boundary_codes(metadata_list: List[BundleMetadata]) -> None:
    """
    Fail-fast guard for the bundle language-code boundary (NEU-133): every
    direction entering a bundle must already declare registry-normalized codes
    ("zh", "zh_hant", bare ISO-639). BCP-47 tags (e.g. "zh-Hans") belong to
    the catalog layer only; a direction carrying one would leak it into the
    bundle manifest and archive name, breaking the deterministic slug rule
    the app's install-time join relies on.

    Args:
        metadata_list (List[BundleMetadata]): Metadata of the bundling inputs.

    Raises:
        ValueError: On the first direction whose codes are not normalized.
    """
    for metadata in metadata_list:
        for field in ("source_language", "target_language"):
            code = metadata[field]
            normalized = normalize_language(code)
            if code != normalized:
                directory = metadata["directory"]
                raise ValueError(
                    f"Direction '{directory}' declares {field} {code!r}, which is "
                    f"not a bundle-boundary code; expected the registry-normalized "
                    f"form {normalized!r} (normalize before bundling — BCP-47 codes "
                    f"live in the catalog layer only, NEU-133)."
                )


def validate_translation_pairs(
    metadata_list: List[BundleMetadata],
) -> None:
    """
    Validates that for each language pair (source -> target), there is a corresponding reverse pair (target -> source).
    Raises on the first set of missing reverse pairs.

    Args:
        metadata_list (List[BundleMetadata]): List of BundleMetadata dictionaries containing source and target language pairs.

    Raises:
        ValueError: If any reverse translation pair is missing.
    """
    # Set to store the existing language pairs
    existing_pairs = set()

    # Populate the existing pairs set
    for metadata in metadata_list:
        existing_pairs.add((metadata["source_language"], metadata["target_language"]))

    # List to store the missing translation pairs
    missing_pairs = []

    # Check for missing reverse translation pairs
    for metadata in metadata_list:
        reverse_pair = (metadata["target_language"], metadata["source_language"])
        if reverse_pair not in existing_pairs:
            missing_pairs.append(reverse_pair)

    if missing_pairs:
        missing_pairs_formatted = [
            f"{source}-{target}" for source, target in missing_pairs
        ]
        raise ValueError(f"Missing translation pairs: {missing_pairs_formatted}")


def extract_unique_languages(metadata: List[BundleMetadata]) -> List[str]:
    """
    Extracts unique languages from a list of language pairs, in the canonical
    bundle-boundary form (normalized, sorted — see versta/language_codes.py).

    Args:
        metadata (List[BundleMetadata]): List of BundleMetadata dictionaries containing source and target language pairs.

    Returns:
        List[str]: A sorted list of unique languages.
    """
    codes: List[str] = []
    for data in metadata:
        codes.extend([data["source_language"], data["target_language"]])
    return bundle_languages(codes)
