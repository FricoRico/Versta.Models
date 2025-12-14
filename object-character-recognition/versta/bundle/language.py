from typing import List

from .typing import OCRBundleMetadata


def extract_unique_languages(metadata_list: List[OCRBundleMetadata]) -> List[str]:
    """
    Extracts unique languages from a list of OCR model metadata.
    If any model contains the wildcard "*", it overrides all other languages.

    Args:
        metadata_list (List[OCRBundleMetadata]): List of OCRBundleMetadata dictionaries.

    Returns:
        List[str]: A list of unique languages, or ["*"] if wildcard is present.
    """
    unique_languages = set()

    # Check for wildcard first
    for metadata in metadata_list:
        languages = metadata.get('languages', [])
        if "*" in languages:
            return ["*"]

    # Collect all unique languages
    for metadata in metadata_list:
        languages = metadata.get('languages', [])
        unique_languages.update(languages)

    # Return the unique languages as a sorted list
    return sorted(unique_languages)


def extract_unique_modules(metadata_list: List[OCRBundleMetadata]) -> List[str]:
    """
    Extracts unique module types from a list of OCR model metadata.

    Args:
        metadata_list (List[OCRBundleMetadata]): List of OCRBundleMetadata dictionaries.

    Returns:
        List[str]: A sorted list of unique module types.
    """
    unique_modules = set()

    for metadata in metadata_list:
        module = metadata.get('module')
        if module:
            unique_modules.add(module)

    # Return the unique modules as a sorted list
    return sorted(unique_modules)

