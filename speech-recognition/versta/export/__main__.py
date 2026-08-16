import os

from argparse import ArgumentParser
from pathlib import Path

from .download import download_model


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Download a whisper.cpp (ggml) speech recognition model from a Hugging Face
        repository and prepare it for bundling. The model file is already in the native
        whisper.cpp format and requires no conversion. A Silero-VAD model is downloaded
        alongside it, as required by the speech-recognition module.
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ggerganov/whisper.cpp",
        help="Hugging Face repository id holding the whisper model. Defaults to 'ggerganov/whisper.cpp'.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="base-q8_0",
        help="Whisper model variant, used to build the filename 'ggml-<model-type>.bin' "
        "(e.g. 'base-q8_0', 'small.en', 'large-v3-turbo-q5_0'). Defaults to 'base-q8_0'.",
    )

    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Supported language codes written to the metadata. Defaults to the full Whisper "
        "set, or ['en'] for English-only ('.en') model variants.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Directory where the downloaded model will be written.",
    )

    return parser.parse_args()


def main(
    model: str,
    model_type: str,
    languages: list,
    output_dir: Path,
) -> Path:
    """
    Downloads a single whisper.cpp model (and its VAD model) and writes it to `output_dir`.

    Args:
        model (str): Hugging Face repository id holding the whisper model.
        model_type (str): Whisper model variant.
        revision (str): Hugging Face repository revision.
        languages (list): Supported language codes (None => default per variant).
        output_dir (Path): Directory where the model will be written.

    Returns:
        Path: The directory containing the downloaded model and its metadata.
    """
    return download_model(
        repo_id=model,
        model_type=model_type,
        output_dir=output_dir,
        languages=languages,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        model_type=args.model_type,
        languages=args.languages,
        output_dir=args.output_dir,
    )
