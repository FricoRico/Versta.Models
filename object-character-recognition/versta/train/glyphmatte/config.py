"""Tunables for the glyphmatte training pipeline.

Every default a pipeline run can be steered with lives here; `__main__.py`'s
argument defaults reference these values so the CLI and any programmatic
use agree.

Also the shared layout of the pipeline's on-disk paths (intermediates are
deleted unless `--keep_intermediates`), and the model's output contract.
"""

from dataclasses import dataclass
from typing import Dict

from ...dataset.glyphmatte.config import DATASET

# The U-Net's named heads, in head order. One source for the exporter, the
# eval harness and the runtime metadata.
OUTPUT_NAMES = ("matte", "weight", "foreground", "background")

# The published dataset snapshot the training pipeline consumes (HF dataset
# repo; 10 train shards + 1 validation shard + metadata.jsonl).
DATASET_REPO = "Neurora/versta-glyphmatte"


@dataclass(frozen=True)
class Layout:
    """Path names under `--output_dir`. Intermediates are removed at the end
    of a run unless `--keep_intermediates`."""

    intermediates: str = "intermediates"
    assets: str = "intermediates/assets"
    dataset: str = "intermediates/dataset"
    ckpt_dir: str = "intermediates/ckpt"
    ckpt_latest: str = "intermediates/ckpt/glyphmatte-latest.pt"
    onnx: str = "intermediates/glyphmatte.onnx"
    eval_json: str = "eval-glyphmatte.json"


LAYOUT = Layout()


@dataclass(frozen=True)
class TrainDefaults:
    """`__main__` argparse defaults."""

    batch: int = 32
    steps: int = 20000
    lr: float = 1e-3
    val: int = 200
    workers: int = 8
    reuse: int = 16
    seed: int = 0
    device: str = "auto"
    degrade_ramp: int = 500
    wrapper_width: int = 512
    size: str = "16x4"
    weight_head: str = "1x1"
    eval_n: int = 64
    val_seed: int = DATASET.seed
    compile: bool = True


TRAIN = TrainDefaults()

# Loss term merge weights. Weight is emphasised: sub-tile strokes carry the
# most per-pixel discrimination signal for the weight head.
LOSS_WEIGHTS: Dict[str, float] = {
    "dice": 1.0,
    "matte_bce": 1.0,
    "weight": 3.0,
    "color": 1.0,
}
