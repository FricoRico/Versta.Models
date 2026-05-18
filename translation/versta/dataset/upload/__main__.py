import os
from argparse import ArgumentParser
from pathlib import Path

from .upload import upload_dataset


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="Upload processed dataset to HuggingFace Hub.",
    )

    parser.add_argument(
        "--name",
        "--dataset-name",
        type=str,
        help="HuggingFace Hub dataset name to push to.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to processed JSONL file(s).",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    upload_dataset(input_paths=args.input, dataset_name=args.name)
