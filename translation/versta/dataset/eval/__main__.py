import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from ..corpus import filter_corpus_config, load_corpus_config

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
from .eval import generate_eval_dataset


def parse_args():
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="Generate a dedicated evaluation dataset with synthetic and natural entries.",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default="corpora.json",
        help="Path to the corpora JSON config file.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source ISO language code (e.g. 'en').",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target ISO language code (e.g. 'nl').",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name of the existing training set "
        "(used to avoid cross-contamination).",
    )

    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Split name of the existing training set. Default: train.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dataset/eval"),
        help="Output directory. Default: output/dataset/eval",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1778419142,
        help="Random seed. Default: 1778419142.",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache"),
        help="Cache directory for OPUS downloads. Default: cache/",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of sentence pairs per LLM batch request. Default: 20.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for LLM inference. Default: 4.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    corpus_configs = load_corpus_config(args.corpus)
    filtered = filter_corpus_config(corpus_configs, args.source, args.target)

    if not filtered:
        print(
            f"No corpus config found for {args.source}-{args.target}. "
            "Check the corpus JSON file."
        )
        return

    config = filtered[0]
    group = config.get("eval")
    if group is None:
        print(
            f"No 'eval' section found in corpus config for "
            f"{args.source}-{args.target}. "
            "Add an 'eval' key to your corpora JSON file."
        )
        return

    source = args.source
    target = args.target
    lang_pair = f"{source}-{target}"
    output_dir = args.output / lang_pair
    download_dir = args.cache / "corpora"
    intermediates_dir = output_dir / "intermediates"

    generate_eval_dataset(
        source=source,
        target=target,
        dataset=args.dataset,
        synthetic_configs=group.get("synthetic", []),
        natural_configs=group.get("natural", []),
        train_split=args.train_split,
        seed=args.seed,
        download_dir=download_dir,
        intermediates_dir=intermediates_dir,
        batch_size=args.batch_size,
        max_workers=args.workers,
        output_path=output_dir / "dataset.jsonl",
    )


if __name__ == "__main__":
    main()
