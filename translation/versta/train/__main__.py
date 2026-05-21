import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from .config import get_dtype
from .dataset import language_pair, load_dataset
from .prune import prune
from .train import finetune, recover
from .utils import remove_folder


def parse_args():
    parser = ArgumentParser(
        description="Fine-tune LFM2.5-350M for English→Dutch translation."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/train"),
        help="Output directory for merged model. Default: output/",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache"),
        help="Cache directory for Unsloth. Default: cache/",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length. Default: 128.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="LiquidAI/LFM2.5-350M-Base",
        help="Model name. Default: LiquidAI/LFM2.5-350M-Base.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Neurora/versta-tonality-en-nl",
        help="Dataset name. Default: Neurora/versta-tonality-en-nl.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Per-device batch size. Default: 256.",
    )

    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.8,
        help="Structured pruning ratio (0.0-1.0). Default: 0.8.",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Whether to remove intermediate files created during the conversion process."
        "This will default to False if not specified.",
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Steps interval for saving and evaluation. Default: 1000.",
    )

    return parser.parse_args()


def main(
    dataset: str,
    output_dir: Path = Path("output/train"),
    cache_dir: Path = Path("cache"),
    model: str = "LiquidAI/LFM2.5-350M-Base",
    max_seq_len: int = 128,
    batch_size: int = 256,
    prune_ratio: float = 0.6,
    keep_intermediates: bool = False,
    save_steps: int = 1000,
) -> None:
    """
    Main training entry point.

    Args:
        dataset: Dataset name.
        output_dir: Output directory for merged model.
        cache_dir: Cache directory for Unsloth.
        model: Model name.
        max_seq_len: Maximum sequence length.
        batch_size: Per-device batch size.
        prune_ratio: Structured pruning ratio.
        keep_intermediates: Whether to keep intermediate files.
        save_steps: Steps interval for saving and evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    source, target = language_pair(dataset)
    lang_pair = f"{source}-{target}"

    intermediates_dir = output_dir / lang_pair / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    os.environ["UNSLOTH_CACHE_DIR"] = cache_dir.as_posix()
    os.environ["UNSLOTH_COMPILE_CACHE_DIR"] = cache_dir.as_posix()

    dataset_data = load_dataset(
        dataset_name=dataset,
        model_name=model,
    )

    tokenizer, merged_path = finetune(
        model=model,
        dataset=dataset_data,
        output_dir=output_dir,
        lang_pair=lang_pair,
        intermediates_dir=intermediates_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_len,
        save_steps=save_steps,
    )

    pruned_dir = prune(
        merged_model_path=merged_path,
        tokenizer=tokenizer,
        prune_ratio=prune_ratio,
        output_dir=intermediates_dir / "pruned",
    )

    recover(
        model=pruned_dir,
        tokenizer=tokenizer,
        dataset=dataset_data,
        output_dir=output_dir,
        lang_pair=lang_pair,
        intermediates_dir=intermediates_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_len,
        cache_dir=cache_dir,
        save_steps=save_steps,
    )

    if not keep_intermediates:
        remove_folder(intermediates_dir)
        print("Intermediates files cleaned.")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset=args.dataset,
        output_dir=args.output,
        cache_dir=args.cache,
        max_seq_len=args.max_seq_len,
        model=args.model,
        batch_size=args.batch_size,
        prune_ratio=args.prune_ratio,
        keep_intermediates=args.keep_intermediates,
        save_steps=args.save_steps,
    )
