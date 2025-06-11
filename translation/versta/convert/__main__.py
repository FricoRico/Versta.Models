
import os

from argparse import ArgumentParser
from pathlib import Path
from requests import head

from huggingface_hub import snapshot_download

from .download import download_model
from .convert_pt import convert_model_to_pt
from .convert_vocabulary import convert_vocabulary
from .convert_sentence_piece import convert_sentence_piece_files
from .metadata import create_decoder_file
from .utils import remove_folder

def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Convert a model from HuggingFace model hub to ONNX format and then to ORT format.
        The converter is intended to be used with MarianMT translation models, but might work with other models as well.
        This function manages the overall workflow from exporting the model to ONNX, saving the tokenizer, and quantizing the model components.
        After the model is quantized, it is converted to ORT format for deployment on ARM devices.
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Provide the name of the pre-trained model to convert."
        "For the moment, only MarianMT translation models are supported.",
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
        "This could be either 'opus' or 'opus-tatoeba' at the moment, defaulting to 'opus-tatoeba'."
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

def repository_exists(repo_name):
    url = f"https://huggingface.co/{repo_name}"
    response = head(url)
    return response.status_code == 200

def main(
    model: str,
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

    model_path = Path(model)

    # Step 1: Download the model files
    if model.startswith("https"):
        model_path = download_model(model, download_dir)

    if repository_exists(model):
        output = snapshot_download(repo_id=model, local_dir_use_symlinks=False)
        model_path = Path(output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist. Please check the model URI.")

    # Step 2: Convert the vocabulary to a format compatible with PyTorch
    convert_vocabulary(model_path)

    # Step 3: Convert the sentence piece files to the expected format
    convert_sentence_piece_files(model_path)

    # Step 4: Create a decoder configuration file if it does not exist
    create_decoder_file(model_path)

    # Step 5: Convert the model to PyTorch format
    convert_model_to_pt(model_path, export_dir, model_format)

    # Step 6: Remove intermediate files if specified
    if keep_intermediates == False:
        remove_folder(download_dir)
        print("Intermediates files cleaned.")

if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        export_dir=args.export_dir,
        temp_dir=args.temp_dir,
        keep_intermediates=args.keep_intermediates,
        model_format=args.model_format,
    )