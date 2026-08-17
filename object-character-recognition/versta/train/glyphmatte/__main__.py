"""One-shot glyphmatte pipeline: dataset (HF snapshot) -> train -> export -> eval -> publish.

All stage logic lives in `pipeline.py` (each stage is a function). Tuning
defaults — batch size, LR, dataset sizes, loss weights, the model shape — live
in `config.py`; the CLI only takes per-run overrides.

Every run keeps its work under `--output_dir`: intermediates in
`intermediates/` (removed at the end unless `--keep_intermediates`), the HF
model-repo upload set at the output root (`onnx/glyphmatte{,_fp16}.onnx`,
`glyphmatte.safetensors`, `config.json`), and the eval report as a JSON at the
root.

The training dataset is the published snapshot from the HF dataset repo
`Neurora/versta-glyphmatte`, downloaded via `huggingface_hub.snapshot_download`
into `intermediates/dataset/` on first use.

    uv sync --extra rocm            # or --extra cu130 on NVIDIA
    uv run python -m versta.train.glyphmatte

After the run: upload the output root (onnx/ + safetensors + config.json) to
the HF model repo `Neurora/versta-glyphmatte` via the HF web UI, then
`uv run python -m versta.export --models glyphmatte` converts the fp32 ONNX
into the pack's int8 MNN.

On ROCm hosts without a C++ toolchain (immutable distros), run inside a
toolbox container: prefix the command with `toolbox run -c glyphmatte`.
"""

import argparse
import os

from argparse import Namespace
from pathlib import Path

# ROCm: silence librocprofiler-register's probing noise ("(null): No such
# file or directory", one per torch-importing process, incl. spawn workers).
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

from . import pipeline
from .config import TRAIN


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/train/glyphmatte"),
        help="Root for intermediates/, the HF upload set and the eval report.",
    )
    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Keep intermediates/ (dataset snapshot, ckpt, onnx) after the run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TRAIN.steps,
        help="Total optimizer steps.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint path (needs --keep_intermediates from a prior run).",
    )
    parser.add_argument(
        "--device",
        default=TRAIN.device,
        help="Torch device: auto/cpu/cuda:N. Default auto prefers cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN.seed,
        help="Training seed (the dataset seed stays fixed at the config value).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline.stage_dataset(args)
    pipeline.stage_train(args)
    pipeline.stage_export(args)
    pipeline.stage_eval(args)
    pipeline.stage_publish(args)
    pipeline.cleanup_intermediates(args)


if __name__ == "__main__":
    main()
