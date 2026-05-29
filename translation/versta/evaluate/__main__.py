import json
import time
from argparse import ArgumentParser
from pathlib import Path

from .evaluate import evaluate
from .types import EvaluationConfig

TONES = ["formal", "neutral", "casual"]


def parse_args():
    parser = ArgumentParser(
        description="Evaluate a translation model using COMET+, BLEU, and chrF metrics."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model name.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="openlanguagedata/flores_plus",
        help="Dataset path or HuggingFace dataset name. Default: openlanguagedata/flores_plus",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source language code (ISO 639-1). Default: en",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target language code (ISO 639-1). Default: nl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluate"),
        help="Output JSON path. Default: output/eval/results_{timestamp}.json",
    )

    parser.add_argument(
        "--percentage",
        type=float,
        default=1.0,
        help="Dataset percentage to use for evaluation (0.0-1.0). Default: 1.0",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Generation batch size. Default: 64",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length. Default: 512",
    )

    parser.add_argument(
        "--tone",
        type=str,
        action="append",
        choices=TONES,
        default=None,
        help="Tone(s) to evaluate. Can be specified multiple times. Default: all tones for custom datasets, ignored for FLORES",
    )

    parser.add_argument(
        "--no-comet",
        action="store_true",
        default=False,
        help="Disable COMET+ scoring.",
    )

    parser.add_argument(
        "--no-bleu",
        action="store_true",
        default=False,
        help="Disable BLEU scoring.",
    )

    parser.add_argument(
        "--no-chrf",
        action="store_true",
        default=False,
        help="Disable chrF scoring.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output / f"{args.source}-{args.target}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tones = args.tone if args.tone else TONES

    config: EvaluationConfig = {
        "model": args.model,
        "dataset": args.dataset,
        "output": output_dir,
        "source": args.source,
        "target": args.target,
        "percentage": args.percentage,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_len,
        "tones": tones,
        "use_comet": not args.no_comet,
        "use_bleu": not args.no_bleu,
        "use_chrf": not args.no_chrf,
        "gen_config": {
            "temperature": 0.1,
            "top_p": 1.0,
            "min_p": 0.1,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.05,
        },
    }

    result = evaluate(config)

    results_path = output_dir / f"results_{int(time.time())}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"Evaluation completed. Results saved to: {results_path}")


if __name__ == "__main__":
    main()
