from pathlib import Path
from typing import List
from json import load

from ..download.download import (
    download_model,
    get_entry,
    load_registry,
)
from ..bundle import __main__ as bundle

from .typing import ModelFile, ExportedBundle, ExportedModel


def export_models(
    models: List[List[ModelFile]],
    output_dir: Path,
    registry_url: str,
) -> List[List[ExportedBundle]]:
    """
    Download the Firefox (Bergamot) translation models and bundle them together.

    Args:
        models (List[List[ModelFile]]): A list of model pairs to be downloaded.
        output_dir (Path): The directory where the models will be downloaded and bundled.
        registry_url (str): URL of the Firefox translations model registry JSON.

    Returns:
        List[List[ExportedBundle]]: A list of dictionaries containing the bundle output details.
    """
    registry = load_registry(registry_url)
    base_url = registry.get("baseUrl", registry_url.rsplit("/", 1)[0])

    exported_bundles: List[List[ExportedBundle]] = []

    for pair in models:
        exported_pair: List[ExportedModel] = []

        for entry in pair:
            architecture = entry.get("architecture")
            registry_entry = get_entry(
                registry,
                entry["source_language"],
                entry["target_language"],
                architecture,
            )

            direction_dir = (
                output_dir / f"{entry['source_language']}-{entry['target_language']}"
            )
            downloaded = download_model(base_url, registry_entry, direction_dir)

            with open(downloaded / "metadata.json", "r") as f:
                metadata = load(f)

            exported_pair.append(
                ExportedModel(
                    path=downloaded,
                    source_language=metadata["source_language"],
                    target_language=metadata["target_language"],
                    architecture=metadata["architecture"],
                    score=metadata["score"],
                    version=metadata.get("version", ""),
                )
            )

        exported_bundles.append(_export_bundle(exported_pair, output_dir))

    return exported_bundles


def _export_bundle(
    model: List[ExportedModel], output_dir: Path
) -> List[ExportedBundle]:
    """
    Bundle the downloaded models into a single tarball.

    Args:
        model (List[ExportedModel]): A list of downloaded models to be bundled.
        output_dir (Path): The directory where the models will be bundled.

    Returns:
        List[ExportedBundle]: A list of dictionaries containing the bundle output details.
    """
    exported_bundles: List[ExportedBundle] = list()

    input_dirs: List[Path] = list()
    for entry in model:
        input_dirs.append(entry["path"])

    exported = bundle.main(
        input_dirs=input_dirs,
        output_dir=output_dir,
        bidirectional=len(input_dirs) > 1,
        keep_intermediates=False,
    )

    for entry in model:
        exported_bundles.append(
            ExportedBundle(
                path=exported["bundle"],
                checksum=exported["checksum"],
                source_language=entry["source_language"],
                target_language=entry["target_language"],
                architecture=entry["architecture"],
                bidirectional=len(input_dirs) > 1,
                score=entry["score"],
                version=entry["version"],
            )
        )

    return exported_bundles
