"""
Language detection and validation utilities for calibration preparation.
"""

from pathlib import Path
from typing import List

from .definitions import languages as language_definitions


def get_language_from_model_name(model_name: str) -> List[str] | None:
    """
    Auto-detect ISO language codes from model name.

    Returns:
        List of ISO language codes (e.g., ["ar", "fa", "ur"])
        or None if model name not recognized
    """
    model_name_lower = model_name.lower()

    if "korean" in model_name_lower:
        return language_definitions["korean"]
    elif "latin" in model_name_lower:
        return language_definitions["latin"]
    elif "eslav" in model_name_lower:
        return language_definitions["eslav"]
    elif "th" in model_name_lower:
        return language_definitions["th"]
    elif "el" in model_name_lower:
        return language_definitions["el"]
    elif "en" in model_name_lower:
        return language_definitions["en"]
    elif "cyrillic" in model_name_lower:
        return language_definitions["cyrillic"]
    elif "arabic" in model_name_lower:
        return language_definitions["arabic"]
    elif "devanagari" in model_name_lower:
        return language_definitions["devanagari"]
    elif "ta" in model_name_lower:
        return language_definitions["ta"]
    elif "te" in model_name_lower:
        return language_definitions["te"]
    else:
        return None


def get_all_supported_languages() -> List[str]:
    """
    Return all unique ISO codes from all model types in definitions.py.

    Returns:
        List of unique two-letter ISO language codes
    """
    all_codes = []
    for lang_list in language_definitions.values():
        all_codes.extend(lang_list)
    return sorted(list(set(all_codes)))


def prepare_language_subdirectories(
    input_dir: Path, expected_languages: List[str]
) -> None:
    """
    Ensure input directory has subdirectories for each expected language.

    Args:
        input_dir: Root directory containing language subdirs
        expected_languages: Expected ISO language codes (e.g., ["ar", "fa", "ur"])
    """
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
    else:
        for lang_code in expected_languages:
            lang_dir = input_dir / lang_code

            "Print warning of missing language directory"
            if not lang_dir.exists():
                print(f"Warning: Missing calibration subdirectory for language '{lang_code}' in '{input_dir}'")

def get_detected_languages(model_name: str, input_dir: Path) -> List[str]:
    """
    Get languages for calibration with full validation.

    Args:
        model_name: PaddleOCR model name/ID
        input_dir: Directory with language subdirectories

    Returns:
        List of ISO language codes

    Raises:
        RuntimeError: If can't detect languages or subdirectories don't match
    """
    detected = get_language_from_model_name(model_name)

    if detected is None:
        all_supported = get_all_supported_languages()
        raise RuntimeError(
            f"Could not auto-detect language type from model: {model_name}\n"
            f"Supported model types: korean, latin, eslav, th, el, en, cyrillic, arabic, devanagari, ta, te\n"
            f"Solutions:\n"
            f"  1. Ensure model name includes language type identifier\n"
            f"  2. Use a model with known language type\n"
            f"  3. Provide calibration images for all supported ISO codes:\n"
            f"     {', '.join(all_supported)}"
        )

    prepare_language_subdirectories(input_dir, detected)

    return detected
