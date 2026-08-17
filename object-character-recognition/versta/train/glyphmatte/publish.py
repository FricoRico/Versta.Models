"""Emitters for the manual HF publish set — no HF client library involved.

Written to the run's output root in the layout the `Neurora/versta-glyphmatte`
model repo expects, so the directory can be dropped into the HF web UI as-is:

  onnx/glyphmatte.onnx        fp32, self-contained (inlined external data)
  onnx/glyphmatte_fp16.onnx   half-precision graph (onnxconverter-common)
  glyphmatte.safetensors      trained weights, no pickle
  config.json                 model dims/heads, training step, eval metrics
"""

import json
import shutil

from pathlib import Path
from typing import Any, Dict

import onnx
import torch
from onnxconverter_common import float16
from safetensors.torch import save_file

from .export_onnx import load_model
from .model import param_count


def emit_fp16_onnx(fp32_path: Path, out: Path) -> Path:
    """Converts the self-contained fp32 ONNX graph to fp16."""
    model = onnx.load(fp32_path)
    fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(fp16, out)
    return out


def emit_safetensors(ckpt: Path, out: Path) -> Dict[str, Any]:
    """Writes the state dict as safetensors; returns the training config."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg: Dict[str, Any] = {
        k: blob.get("config", {}).get(k) for k in ("base", "levels", "weight_head")
    }
    cfg["step"] = int(blob.get("step", 0))
    sd = {k: v.contiguous() for k, v in blob["state_dict"].items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, out)
    return cfg


def emit_publish_tree(
    ckpt: Path, onnx: Path, out: Path, eval_json: Path | None = None
) -> Path:
    """Builds the full publish tree; returns its root.

    Args:
        ckpt (Path): Training checkpoint (glyphmatte-latest.pt).
        onnx (Path): The exported fp32 ONNX.
        out (Path): Publish dir root (mirrors the HF repo layout).
        eval_json (Path | None): eval results to fold into config.json.
    """
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, onnx_dir / "glyphmatte.onnx")
    emit_fp16_onnx(onnx, onnx_dir / "glyphmatte_fp16.onnx")
    cfg = emit_safetensors(ckpt, out / "glyphmatte.safetensors")
    model = load_model(ckpt)
    cfg["params"] = param_count(model)
    cfg["outputs"] = ["matte", "weight", "foreground", "background"]
    cfg["input_height"] = 48
    if eval_json and eval_json.exists():
        cfg["eval"] = json.loads(eval_json.read_text())
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"publish tree ready: {out}")
    return out
