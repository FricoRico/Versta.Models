from pathlib import Path
from typing import List
from json import load

from ..export import __main__ as export
from ..bundle import __main__ as bundle

from .typing import ModelFile, ExportedBundle, ExportedModel

def export_models(model: List[List[ModelFile]], output_dir: Path, keep_intermediates: bool, clear_cache: bool) -> List[List[ExportedBundle]]:
    """
    Export the models to a specified directory.

    Args:
        model (List[List[ModelFile]]): A list of model pairs to be exported.
        output_dir (Path): The directory where the models will be exported.
        keep_intermediates (bool): Whether to keep intermediate files.
        clear_cache (bool): Whether to clear the cache after exporting.

    Returns:
        List[List[ExportedBundle]]: A list of dictionaries containing the export output details.
    """
    exported_bundles: List[List[ExportedBundle]] = []

    for pair in model:
        exported_pair: List[ExportedModel] = []

        for entry in pair:
            exported = export.main(
                model=entry["base_model"],
                output_dir=output_dir,
                score=entry["score"],
                keep_intermediates=keep_intermediates,
                clear_cache=clear_cache,
            )

            with open(exported["metadata"], "r") as f:
                metadata = load(f)

            exported_pair.append(
                ExportedModel(
                    path=exported["path"],
                    base_model=entry["base_model"],
                    source_language=metadata["source_language"],
                    target_language=metadata["target_language"],
                    architectures=metadata["architectures"],
                    score=entry["score"],
                    version=metadata["version"],
                )
            )

        exported_bundles.append(_export_bundle(exported_pair, output_dir, keep_intermediates))

    return exported_bundles

def _export_bundle(model: List[ExportedModel], output_dir: Path, keep_intermediates: bool) -> List[ExportedBundle]:
    """
    Export the models to a specified directory.
    Args:
        model (List[ExportedModel]): A list of exported models to be bundled.
        output_dir (Path): The directory where the models will be bundled.
        keep_intermediates (bool): Whether to keep intermediate files.

    Returns:
        List[ExportedBundle]: A List of dictionaries containing the bundle output details.
    """
    exported_bundles: List[ExportedBundle] = list()

    input_dirs: List[Path] = list()
    for entry in model:
        input_dirs.append(entry["path"])

    exported = bundle.main(
        input_dirs=input_dirs,
        output_dir=output_dir,
        bidirectional=len(input_dirs) > 1,
        keep_intermediates=keep_intermediates,
    )

    for entry in model:
        exported_bundles.append(
            ExportedBundle(
                path=exported["bundle"],
                checksum=exported["checksum"],
                base_model=entry["base_model"],
                bidirectional=len(input_dirs) > 1,
                architectures=entry["architectures"],
                source_language=entry["source_language"],
                target_language=entry["target_language"],
                score=entry["score"],
                version=entry["version"],
            )
        )

    return exported_bundles