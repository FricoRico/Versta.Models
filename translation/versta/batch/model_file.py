import json
import shutil
from json import load
from os.path import getsize
from pathlib import Path
from typing import Dict, List, Tuple

from ..download.download import normalize_language
from .typing import ExportedBundle, ModelFile


# Fields copied verbatim from the generated models.json into the catalog.
COPIED_FIELDS = ("size", "bundle", "checksum")


def load_model_file(file_path: Path) -> List[List[ModelFile]]:
    """
    Load a model file from the specified path and return its models as a dictionary.

    Each entry describes a single translation direction and is identified by its source and target
    language together with the desired Firefox model architecture (e.g. "tiny").

    Args:
        file_path (str): Path to the model file.

    Returns:
        List[List[ModelFile]]: A list of language pairs, each containing one or more model definitions.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    model_files: List[List[ModelFile]] = list()

    with open(file_path, "r") as f:
        models = load(f)

        for pairs in models:
            model_pairs: List[ModelFile] = list()

            for model in pairs:
                model_pairs.append(
                    ModelFile(
                        source_language=model["source_language"],
                        target_language=model["target_language"],
                        architecture=model.get("architecture") or None,
                        score=float(model.get("score", 0.0)),
                    )
                )

            model_files.append(model_pairs)

    return model_files


def save_model_file(
    bundles: List[List[ExportedBundle]],
    link_prefix: str,
    output_dir: Path,
    version: str,
) -> Path:
    """
    Save the model file to the specified path.

    Args:
        bundles (List[List[ExportedBundle]]): List of model bundles to be saved.
        link_prefix (str): Prefix for the model file links.
        output_dir (Path): Directory where the model file will be saved.

    Returns:
        None
    """
    file_path = output_dir / "models.json"
    model_output: dict = {
        "version": version,
        "models": list(),
    }

    for pairs in bundles:
        model_pairs: List[ModelFile] = list()

        for bundle in pairs:
            model_pairs.append(
                ModelFile(
                    source_language=bundle["source_language"],
                    target_language=bundle["target_language"],
                    architecture=bundle["architecture"],
                    score=bundle["score"],
                    version=bundle["version"],
                    size=getsize(bundle["path"]),
                    bundle=link_prefix + bundle["path"].name,
                    checksum=link_prefix + bundle["checksum"].name,
                )
            )

        model_output["models"].append(model_pairs)

    with open(file_path, "w") as f:
        json.dump(model_output, f, indent=4)

    return file_path


def _entry_key(source_language: str, target_language: str) -> Tuple[str, str]:
    return (normalize_language(source_language), normalize_language(target_language))


def update_models_json(existing_path: Path, generated_path: Path, version: str) -> Path:
    """
    Refreshes the computed fields of an existing catalog models.json from the freshly generated
    models.json, matching entries by (source_language, target_language). For bidirectional entries
    that have no exact match, the reversed language pair is also tried.

    The following catalog fields are updated:
      * version - set to `version` (the deployment version from version.txt) for every entry.
      * size, bundle, checksum - copied verbatim from the generated models.json (matched entries).
      * score - the generated COMET-22 score (0-1) is converted to the catalog's 0-100 scale
        (value * 100, rounded to one decimal) for matched entries.
    Descriptive fields (base_model, architectures, bidirectional, source/target language) are
    preserved unchanged.

    Before overwriting, the existing catalog is backed up to "<input_name>.bak" (any prior backup
    is overwritten).

    Args:
        existing_path (Path): Path to the existing catalog models.json to update in place.
        generated_path (Path): Path to the freshly generated models.json carrying the computed fields.
        version (str): The deployment version (from version.txt) written to every catalog entry.

    Returns:
        Path: The path of the updated catalog (same as `existing_path`).
    """
    existing_path = Path(existing_path)
    generated_path = Path(generated_path)

    backup_path = existing_path.parent / (existing_path.name + ".bak")
    shutil.copy2(existing_path, backup_path)

    with open(existing_path, "r") as f:
        catalog = load(f)
    with open(generated_path, "r") as f:
        generated = load(f)

    computed_by_key: Dict[Tuple[str, str], dict] = {}
    for group in generated.get(
        "models", generated if isinstance(generated, list) else []
    ):
        for entry in group:
            computed_by_key[
                _entry_key(entry["source_language"], entry["target_language"])
            ] = entry

    def lookup(source_language: str, target_language: str, bidirectional: bool):
        key = _entry_key(source_language, target_language)
        if key in computed_by_key:
            return computed_by_key[key]
        if bidirectional:
            reverse = _entry_key(target_language, source_language)
            if reverse in computed_by_key:
                return computed_by_key[reverse]
        return None

    for group in catalog:
        for entry in group:
            entry["version"] = version

            match = lookup(
                entry["source_language"],
                entry["target_language"],
                bool(entry.get("bidirectional")),
            )
            if match is None:
                continue

            for field in COPIED_FIELDS:
                if field in match:
                    entry[field] = match[field]

            if "score" in match:
                entry["score"] = round(float(match["score"]) * 100, 1)

    with open(existing_path, "w") as f:
        json.dump(catalog, f, indent=4)

    return existing_path
