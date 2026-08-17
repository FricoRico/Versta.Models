"""Pipeline stages for the glyphmatte one-shot trainer.

Each stage is a small function so the `__main__.main()` orchestrator stays a
flat sequence. Everything the stage produces lands under
``<output_dir>/intermediates/`` and is removed at the end of the pipeline
unless `--keep_intermediates`; the HF model-repo upload set lands at the output
root (``onnx/``, ``glyphmatte.safetensors``, ``config.json``).

The manual hand-off to `versta.export`: upload the output root to the HF model
repo (see README), then `uv run python -m versta.export --models glyphmatte`
converts the fp32 ONNX to the pack's int8 MNN.
"""

import argparse
import json
import shutil

from pathlib import Path

from dataclasses import replace

from huggingface_hub import snapshot_download

from .config import DATASET_REPO, LAYOUT, TRAIN
from .export_onnx import export, load_model
from .train import train_model


def stage_dataset(args: argparse.Namespace) -> Path:
    """Downloads the published dataset snapshot from the HF hub.

    Skipped when shards already exist locally (repeat runs stay fully
    local). Returns the dataset dir.
    """
    ds_dir = Path(args.output_dir) / LAYOUT.dataset
    if (ds_dir / "metadata.jsonl").exists() and list(ds_dir.glob("train-*.parquet")):
        print(f"dataset already present: {ds_dir}")
        return ds_dir
    snapshot = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["metadata.jsonl", "*.parquet"],
        local_dir=ds_dir,
    )
    shards = len(list(ds_dir.glob("train-*.parquet")))
    print(f"dataset snapshot: {snapshot} ({shards} train shards)")
    return ds_dir


def stage_train(args: argparse.Namespace) -> None:
    cfg = replace(TRAIN, steps=args.steps, device=args.device, seed=args.seed)
    train_model(cfg, Path(args.output_dir), args.resume)


def stage_export(args: argparse.Namespace) -> Path:
    """Exports the latest checkpoint to a self-contained fp32 ONNX."""
    ckpt = args.resume or (Path(args.output_dir) / LAYOUT.ckpt_latest)
    onnx_path = Path(args.output_dir) / LAYOUT.onnx
    model = load_model(ckpt)
    export(model, onnx_path)
    return onnx_path


def stage_eval(args: argparse.Namespace) -> None:
    from .eval import evaluate

    results = evaluate(
        ckpt=(args.resume or Path(args.output_dir) / LAYOUT.ckpt_latest),
        onnx=Path(args.output_dir) / LAYOUT.onnx,
        n=TRAIN.eval_n,
        dataset_dir=Path(args.output_dir) / LAYOUT.dataset,
    )
    out = Path(args.output_dir) / LAYOUT.eval_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"eval written: {out}")


def stage_publish(args: argparse.Namespace) -> Path:
    """Emits the manual-upload set at the output root: onnx/{fp32,fp16},
    safetensors, config — the layout the HF model repo expects."""
    from .publish import emit_publish_tree

    return emit_publish_tree(
        ckpt=(args.resume or Path(args.output_dir) / LAYOUT.ckpt_latest),
        onnx=Path(args.output_dir) / LAYOUT.onnx,
        out=Path(args.output_dir),
        eval_json=Path(args.output_dir) / LAYOUT.eval_json,
    )


def cleanup_intermediates(args: argparse.Namespace) -> None:
    if args.keep_intermediates:
        print(f"keeping {args.output_dir}/intermediates (--keep_intermediates)")
        return
    shutil.rmtree(Path(args.output_dir) / LAYOUT.intermediates, ignore_errors=True)
    print("intermediates removed (finals at the output root + eval json kept)")
