import json

from pathlib import Path
from typing import List

from pycountry import languages

from .typing import ORTFiles, TokenizerFiles
from .definitions import languages as languageDefinitions


def generate_metadata(version: str, output_dir: Path, model: str, module: str, ort_files: ORTFiles, tokenizer_files: TokenizerFiles) -> Path:
    """
    Generates a metadata file for the model conversion process.

    Args:
        version (str): Version of the model conversion process.
        model (str): Name of the model being converted.
        module (str): Format of the model ("detector" or "recognizer").
        output_dir (Path): Path to the directory where the metadata file will be saved.
        ort_files (ORTFiles): Dictionary containing the file paths for the encoder and decoder ORT files.
        tokenizer_files (TokenizerFiles): Dictionary containing the file paths for the tokenizer files.
    """
    languageCodes = _get_language_from_model_name(model)
    if languageCodes is None:
        raise ValueError(f"Could not extract language codes from model name: {model}")

    modelName = model.split('/').pop().replace('_', '-').lower()

    metadata = {
        "id": f"{modelName}-{module}",
        "version": version,
        "base_model": model,
        "architectures": ["PaddleOCR"], # Only supported model for now
        "languages": languageCodes,
        "module": module,
        "files": {
            "inference": ort_files or {},
            "tokenizer": tokenizer_files or {}
        }
    }

    # Define the path for the metadata.json file
    metadata_file = output_dir / "metadata.json"

    # Write the metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file

def _get_language_from_model_name(model_name: str) -> List[str]:
    """
    Extracts the language code from the model name.

    Args:
        model_name (str): The name of the model.
    Returns:
        languages (List[str]): A list of extracted language codes.
    """
    if "korean" in model_name.lower():
        definitions = languageDefinitions["korean"]
        return _get_iso_code(definitions)
    elif "latin" in model_name.lower():
        definitions = languageDefinitions["latin"]
        return _get_iso_code(definitions)
    elif "eslav" in model_name.lower():
        definitions = languageDefinitions["eslav"]
        return _get_iso_code(definitions)
    elif "th" in model_name.lower():
        definitions = languageDefinitions["th"]
        return _get_iso_code(definitions)
    elif "el" in model_name.lower():
        definitions = languageDefinitions["el"]
        return _get_iso_code(definitions)
    elif "en" in model_name.lower():
        definitions = languageDefinitions["en"]
        return _get_iso_code(definitions)
    elif "cyrillic" in model_name.lower():
        definitions = languageDefinitions["cyrillic"]
        return _get_iso_code(definitions)
    elif "arabic" in model_name.lower():
        definitions = languageDefinitions["arabic"]
        return _get_iso_code(definitions)
    elif "devanagari" in model_name.lower():
        definitions = languageDefinitions["devanagari"]
        return _get_iso_code(definitions)
    elif "ta" in model_name.lower():
        definitions = languageDefinitions["ta"]
        return _get_iso_code(definitions)
    elif "te" in model_name.lower():
        definitions = languageDefinitions["te"]
        return _get_iso_code(definitions)
    else:
        return ["*"]

def _get_iso_code(language_names: List[str]) -> List[str]:
    """
    Return ISO 639-1 codes for the provided language list, keeping failures as None.

    Args:
        language_names: Iterable of language descriptions (e.g., "Serbian (Latin)").

    Returns:
        Mapping from the original language description to its ISO 639-1 code or None if unavailable.
    """
    results: List[str] = []

    for raw in language_names:
        cleaned = raw.split("(")[0].strip()
        try:
            record = languages.get(name = cleaned)
        except LookupError:
            raise ValueError(f"Could not extract language codes from language name: {raw}")
        else:
            results.append(getattr(record, "alpha_2", None))

    return [code for code in results if code is not None]