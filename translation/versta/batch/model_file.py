import json
from os.path import getsize
from pathlib import Path
from typing import List
from json import load

from .typing import ExportedBundle, ModelFile

def load_model_file(file_path: Path) -> List[List[ModelFile]]:
    """
    Load a model file from the specified path and return its models as a dictionary.

    Args:
        file_path (str): Path to the model file.

    Returns:
        List[List[ModelFile]]: A dictionary containing the model name and its score.
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
                    ModelFile(base_model=model["base_model"], score=float(model["score"]))
                )

            model_files.append(model_pairs)

    return model_files

def save_model_file(bundles: List[List[ExportedBundle]], link_prefix: str, output_dir: Path) -> Path:
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

    model_output: List[List[ModelFile]] = list()
    for pairs in bundles:
        model_pairs: List[ModelFile] = list()

        for bundle in pairs:
            model_pairs.append(
                ModelFile(
                    base_model=bundle["base_model"],
                    source_language=bundle["source_language"],
                    target_language=bundle["target_language"],
                    bidirectional=bundle["bidirectional"],
                    architectures=bundle["architectures"],
                    score=bundle["score"],
                    version=bundle["version"],
                    size=getsize(bundle["path"]),
                    bundle=link_prefix + bundle["path"].name,
                    checksum=link_prefix + bundle["checksum"].name,
                )
            )

        model_output.append(model_pairs)


    with open(file_path, "w") as f:
        json.dump(model_output, f, indent=4)

    return file_path