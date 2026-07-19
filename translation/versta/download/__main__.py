import os

from argparse import ArgumentParser
from pathlib import Path

from .download import (
    download_model,
    get_entry,
    load_registry,
)

def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Download a Firefox (Bergamot) translation model for a single language direction
        from Mozilla's public Google Cloud Storage bucket and prepare it for bundling.
        The model files are already in the native Bergamot format and require no conversion.
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source language code (e.g. 'en').",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target language code (e.g. 'nl').",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="tiny",
        choices=["tiny", "base", "base-memory"],
        help="Model architecture to download. Defaults to 'tiny'.",
    )

    parser.add_argument(
        "--registry_url",
        type=str,
        default="https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json",
        help="URL of the Firefox translations model registry JSON.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Directory where the downloaded model will be written.",
    )

    return parser.parse_args()


def main(
        source: str,
        target: str,
        architecture: str,
        registry_url: str,
        output_dir: Path,
) -> Path:
    """
    Downloads a single Firefox translation model direction and writes it to `output_dir`.

    Args:
        src (str): Source language code.
        tgt (str): Target language code.
        architecture (str): Model architecture ("tiny", "base" or "base-memory").
        registry_url (str): URL of the model registry JSON.
        output_dir (Path): Directory where the model will be written.

    Returns:
        Path: The directory containing the downloaded model and its metadata.
    """
    registry = load_registry(registry_url)
    base_url = registry.get("baseUrl", registry_url.rsplit("/", 1)[0])
    entry = get_entry(registry, source, target, architecture)

    direction_dir = Path(output_dir) / f"{source}-{target}"
    return download_model(base_url, entry, direction_dir)


if __name__ == "__main__":
    args = parse_args()
    main(
        source=args.source,
        target=args.target,
        architecture=args.architecture,
        registry_url=args.registry_url,
        output_dir=args.output_dir,
    )
