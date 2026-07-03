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
        type=str,
        required=True,
        help="HuggingFace Hub dataset name to push to.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to processed JSONL file(s).",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="What dataset split to target.",
    )

    return parser.parse_args()


def main(input_paths: list[Path], dataset_name: str, split: str = "train") -> None:
    upload_dataset(input_paths=input_paths, dataset_name=dataset_name, split=split)


if __name__ == "__main__":
    args = parse_args()
    main(input_paths=args.input, dataset_name=args.name, split=args.split)
