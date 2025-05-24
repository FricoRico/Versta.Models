import os

from argparse import ArgumentParser
from pathlib import Path

from .model_file import load_model_file, save_model_file
from .export import export_models

def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Batch convert and bundle multiple language models together and generate an output definition,
        which can be used to deploy the models in the Versta application. This will allow the app to easily download
        the models from the cloud object storage.
        """,
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        help="Provie the file containing the HuggingFace model names to convert. "
             "This JSON file will be used to download the models from HuggingFace and convert them to ONNX format.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Provide an output directory for the converted model's and configuration file. "
             "If unspecified, the converted ORT format model's will be in the '/output' directory.",
    )

    parser.add_argument(
        "--link_prefix",
        type=str,
        default="https://models.versta.app/translation/",
        help="Provide the prefix for the links to the models. "
             "This will be used to generate the links to the models in the output definition file.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
             "This will default to False if not specified.",
    )

    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=False,
        help="Whether to remove the downloaded files from HuggingFace cache."
             "This will default to False if not specified.",
    )

    parsed_args = parser.parse_args()
    return parsed_args

def main(
        input_file: Path,
        output_dir: Path,
        link_prefix: str,
        keep_intermediates: bool = False,
        clear_cache: bool = False,
):
    # Step 1: Load the model file
    models = load_model_file(input_file)

    # Step 2: Export all models to ONNX format and bundles them together
    bundles = export_models(models, output_dir, keep_intermediates, clear_cache)

    # Step 3: Save the model file
    save_model_file(bundles, link_prefix, output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        link_prefix=args.link_prefix,
        keep_intermediates=args.keep_intermediates,
        clear_cache=args.clear_cache,
    )