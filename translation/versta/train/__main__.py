from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree, ignore_patterns

from dotenv import load_dotenv

from .dataset import language_pair, load_dataset
from .prune import prune
from .train import finetune, recover
from .utils import remove_folder

# Load .env for HF_TOKEN (auto-picked by HuggingFace SDK)
load_dotenv(Path(__file__).parent.parent.parent / ".env")


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
        default="LiquidAI/LFM2.5-230M-Base",
        help="Model name. Default: LiquidAI/LFM2.5-230M-Base.",
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
        default=64,
        help="Per-device batch size. Default: 256.",
    )

    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.48,
        help="Structured pruning ratio (0.0-1.0). Default: 0.48.",
    )

    parser.add_argument(
        "--enable-pruning",
        action="store_true",
        default=False,
        help="Wether to prune the model and recovery train it afterwards. Since the release of LiquidAI/LFM2.5-230M-Base, this will no longer be required.",
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
        default=10000,
        help="Steps interval for saving and evaluation. Default: 10000.",
    )

    return parser.parse_args()


def main(
    dataset: str,
    output_dir: Path = Path("output/train"),
    cache_dir: Path = Path("cache"),
    model: str = "LiquidAI/LFM2.5-350M-Base",
    max_seq_len: int = 128,
    batch_size: int = 64,
    prune_ratio: float = 0.48,
    enable_pruning: bool = False,
    keep_intermediates: bool = False,
    save_steps: int = 10000,
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

    language_output_dir = output_dir / lang_pair
    intermediates_dir = language_output_dir / "intermediates"
    finetuned_dir = intermediates_dir / "finetuned"
    pruned_dir = intermediates_dir / "pruned"
    logs_dir = intermediates_dir / "logs"

    language_output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    pruned_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_data = load_dataset(
        dataset_name=dataset,
        model_name=model,
    )

    finetuned = finetune(
        model=model,
        dataset=dataset_data,
        output_dir=finetuned_dir,
        lang_pair=lang_pair,
        batch_size=batch_size,
        max_seq_length=max_seq_len,
        save_steps=save_steps,
        logs_dir=logs_dir,
    )

    if enable_pruning:
        pruned = prune(
            model=finetuned,
            prune_ratio=prune_ratio,
            output_dir=pruned_dir,
        )

        recover(
            model=pruned,
            dataset=dataset_data,
            output_dir=language_output_dir,
            intermediates_dir=intermediates_dir,
            lang_pair=lang_pair,
            logs_dir=logs_dir,
            batch_size=batch_size,
            max_seq_length=max_seq_len,
            save_steps=save_steps,
        )
    else:
        copytree(
            src=finetuned_dir,
            dst=output_dir,
            ignore=ignore_patterns("checkpoints", "adapter"),
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
        enable_pruning=args.enable_pruning,
        keep_intermediates=args.keep_intermediates,
        save_steps=args.save_steps,
    )
