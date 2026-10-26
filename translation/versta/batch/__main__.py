import os

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional

from .model_file import load_model_file, save_model_file, update_models_json
from .export import export_models
from .publish_gate import (
    DEFAULT_BANDS_FILE,
    DEFAULT_SCORECARDS_DIR,
    gate_bundles,
)

with open(Path(__file__).parent.parent / "version.txt", "r") as _version_file:
    BUNDLE_VERSION = _version_file.read().strip()


def parse_args() -> Namespace:
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Batch download and bundle multiple Firefox (Bergamot) translation models together
        and generate an output definition, which can be used to deploy the models in the Versta application.
        This will allow the app to easily download the models from the cloud object storage.
        """,
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        help="Provide the file containing the Firefox language pairs to download. "
        "This JSON file will be used to download the models from Mozilla's storage bucket.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/export"),
        help="Provide an output directory for the downloaded models and configuration file. "
        "If unspecified, the models will be saved in the './output' directory.",
    )

    parser.add_argument(
        "--link_prefix",
        type=str,
        default="https://models.versta.app/translation/",
        help="Provide the prefix for the links to the models. "
        "This will be used to generate the links to the models in the output definition file.",
    )

    parser.add_argument(
        "--registry_url",
        type=str,
        default="https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json",
        help="URL of the Firefox translations model registry JSON.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the bundling process."
        "This will default to False if not specified.",
    )

    parser.add_argument(
        "--scorecards_dir",
        type=Path,
        default=DEFAULT_SCORECARDS_DIR,
        help="Directory of pack-acceptance scorecards consumed by the publish gate "
        "(NEU-60). A bundle may not publish without a passing card for its exact bytes.",
    )

    parser.add_argument(
        "--bands_file",
        type=Path,
        default=DEFAULT_BANDS_FILE,
        help="Pinned golden-output bands (pack-acceptance/bands.json) the publish "
        "gate checks chrF and digit preservation against.",
    )

    parser.add_argument(
        "--skip_publish_gate",
        action="store_true",
        default=False,
        help="Skip the publish gate (NEU-60) for development/dry-run exports only: "
        "the produced bundles may then have no scorecard at all. Never set this "
        "when producing artifacts meant for models.versta.app.",
    )

    parsed_args = parser.parse_args()
    return parsed_args


def main(
    input_file: Path,
    output_dir: Path,
    link_prefix: str,
    registry_url: str,
    keep_intermediates: bool = False,
    scorecards_dir: Optional[Path] = None,
    bands_file: Path = DEFAULT_BANDS_FILE,
    skip_publish_gate: bool = False,
) -> None:
    # Step 1: Load the model file
    models = load_model_file(input_file)

    # Step 2: Download all models and bundle them together
    bundles = export_models(models, output_dir, registry_url)

    # Publish gate (NEU-60): every produced bundle must be backed by a
    # passing pack-acceptance scorecard for its exact bytes, inside the
    # pinned golden bands — no bundle ships unproven. The gate runs before
    # any catalog artifact is written, so a refusal leaves the published
    # catalog and the git-tracked input file untouched.
    if skip_publish_gate:
        print(
            "publish gate SKIPPED (--skip_publish_gate): development/dry-run "
            "export only; the produced bundles are not publishable evidence."
        )
    else:
        produced: List[Path] = sorted(
            {Path(entry["path"]) for pair in bundles for entry in pair}
        )
        gate_bundles(
            produced,
            scorecards_dir or DEFAULT_SCORECARDS_DIR,
            bands_file,
        )
        print(
            f"publish gate: all {len(produced)} bundle(s) backed by passing scorecards"
        )

    # Step 3: Save the model file
    save_model_file(bundles, link_prefix, output_dir, BUNDLE_VERSION)

    # Step 4: Sync the freshly computed fields back into the input catalog (backed up to .bak)
    update_models_json(input_file, output_dir / "models.json", BUNDLE_VERSION)


if __name__ == "__main__":
    args = parse_args()
    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        link_prefix=args.link_prefix,
        registry_url=args.registry_url,
        keep_intermediates=args.keep_intermediates,
        scorecards_dir=args.scorecards_dir,
        bands_file=args.bands_file,
        skip_publish_gate=args.skip_publish_gate,
    )
