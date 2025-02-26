
import os

from argparse import ArgumentParser
from pathlib import Path

from .download import download_model
from .convert_pt import convert_model_to_pt
from .utils import remove_folder

def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Convert a model from HuggingFace model hub to ONNX format and then to ORT format.
        The converter is intended to be used with Helsinki-NLP's Opus-MT translation models, but might work with other models as well.
        This function manages the overall workflow from exporting the model to ONNX, saving the tokenizer, and quantizing the model components.
        After the model is quantized, it is converted to ORT format for deployment on ARM devices.
        """,
    )

    parser.add_argument(
        "--model_uri",
        type=str,
        help="Provide the name of the pre-trained model to convert."
        "For the moment, only Helsinki-NLP's Opus-MT translation models are supported.",
        required=True,
    )

    parser.add_argument(
        "--export_dir",
        type=Path,
        default=Path("export"),
        help="Provide an output directory for the converted model's and configuration file."
        "If unspecified, the converted ORT format model's will be in the '/output' directory, in the provided language.",
    )

    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=Path("tmp"),
        help="Provide an temporary directory used to download and extract raw models."
        "If unspecified, the downloaded models will go into '/tmp' directory.",
    )

    parser.add_argument(
        "--model_format",
        type=str,
        default="opus-tatoeba",
        help="Specify the format of the model to convert."
        "This could be either 'opus' or 'opus-tatoeba' at the moment, defaulting to 'opus'."
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
        "This will default to False if not specified.",
    )

    parsed_args = parser.parse_args()
    return parsed_args

def main(
    model_uri: str,
    export_dir: Path,
    temp_dir: Path,
    keep_intermediates: bool = False,
    model_format: str = "opus",
):
    """
    Main function to handle the model conversion process.
    This function manages the overall workflow for converting an OpusMT model to Torch.
    """
    download_dir = temp_dir / "downloads"

    download_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Check if the model is a direct download
    if model_uri.startswith("https") == False:
        raise ValueError("The model URI must be a direct download link.")

    # Step 1: Download the model files
    downloaded_files = download_model(model_uri, download_dir)

    # Step 2: Convert the model to PyTorch format
    convert_model_to_pt(downloaded_files, export_dir, model_format)

    # Step 3: Remove intermediate files if specified
    if keep_intermediates == False:
        remove_folder(download_dir)
        print("Intermediates files cleaned.")

if __name__ == "__main__":
    args = parse_args()
    main(
        model_uri=args.model_uri,
        export_dir=args.export_dir,
        temp_dir=args.temp_dir,
        keep_intermediates=args.keep_intermediates,
        model_format=args.model_format,
    )