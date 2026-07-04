import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from ..corpus import filter_corpus_config, load_corpus_config
from ..types import CorpusGroupConfig
from .eval import generate_eval_dataset

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


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
        "--split",
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


def main(
    group: CorpusGroupConfig,
    source: str,
    target: str,
    dataset: str,
    split: str = "train",
    seed: int = 1778419142,
    output: Path = Path("output/dataset/eval"),
    cache: Path = Path("cache"),
    batch_size: int = 20,
    workers: int = 16,
) -> None:
    lang_pair = f"{source}-{target}"
    output_dir = output / lang_pair
    download_dir = cache / "corpora"
    intermediates_dir = output_dir / "intermediates"

    download_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_eval_dataset(
        source=source,
        target=target,
        dataset=dataset,
        synthetic_configs=group.get("synthetic", []),
        natural_configs=group.get("natural", []),
        split=split,
        seed=seed,
        download_dir=download_dir,
        intermediates_dir=intermediates_dir,
        batch_size=batch_size,
        max_workers=workers,
        output_path=output_dir / "dataset.jsonl",
    )


if __name__ == "__main__":
    args = parse_args()

    corpus_configs = load_corpus_config(args.corpus)
    corpus_configs = filter_corpus_config(corpus_configs, args.source, args.target)

    group = corpus_configs[0].get("eval")

    main(
        group=group,
        source=args.source,
        target=args.target,
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        output=args.output,
        cache=args.cache,
        batch_size=args.batch_size,
        workers=args.workers,
    )
