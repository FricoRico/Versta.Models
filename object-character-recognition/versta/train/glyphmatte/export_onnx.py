"""Export the trained glyph-matte U-Net to ONNX (dynamic strip width).

Reference: MIT-licensed translator-rs/scripts/ink_model/export_onnx.py by
David Ventura (approach port).

  https://github.com/DavidVentura/translator-rs/blob/master/scripts/ink_model/export_onnx.py

Dummy input 1x3x48x320, input name `strip`, outputs `matte`, `weight`,
`foreground`, `background`; width dynamic on all. Verifies torch-vs-ORT (CPU)
max|delta| < 1e-3 at widths 160/320/504 — catches dynamic-width and opset
mistakes before MNN conversion.

CLI: `uv run python -m versta.train.glyphmatte.export_onnx ckpt/glyphmatte-latest.pt out.onnx`
"""

import sys

from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .gen_data import HEIGHT
from .model import CompatUNet, GlyphMatteUNet

OUTPUT_NAMES = ["matte", "weight", "foreground", "background"]


def load_model(
    ckpt: Path, device: torch.device = torch.device("cpu")
) -> GlyphMatteUNet:
    """Loads a checkpoint back into an eager GlyphMatteUNet.

    Accepts the pre-color 2-head checkpoints: missing color weights are
    re-initialised, `bold_head.*` keys are remapped to `weight_head.*`.

    Args:
        ckpt (Path): Checkpoint written by `train.save_ckpt`.
        device (torch.device): Target device, CPU for export.

    Returns:
        GlyphMatteUNet: The model in eval mode.
    """
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg: Dict[str, object] = blob.get("config", {})
    model = GlyphMatteUNet(
        base=int(cfg.get("base", 16)),
        levels=int(cfg.get("levels", 4)),
        weight_head=str(cfg.get("weight_head", cfg.get("bold_head", "1x1"))),
    )
    sd = blob["state_dict"]
    if not any(k.startswith("weight_head") for k in sd):
        sd = CompatUNet.remap_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected ckpt keys: {unexpected}")
    if missing:
        # Tolerated only for the color head on legacy 2-head checkpoints.
        bad = [k for k in missing if not k.startswith("color_head")]
        if bad:
            raise RuntimeError(f"missing ckpt keys: {bad}")
        print(f"legacy ckpt: colour head re-initialised ({len(missing)} missing)")
    return model.eval().to(device)


def export(model: GlyphMatteUNet, out: Path) -> None:
    """Writes the ONNX graph with dynamic width; validates graph + parity.

    Args:
        model (GlyphMatteUNet): The eval-mode model.
        out (Path): Output .onnx path.

    Raises:
        RuntimeError: Graph invalid, wrong shapes or torch/ORT divergence.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, HEIGHT, 320)

    class _TupleOut(torch.nn.Module):
        """Dynamo exporter needs an nn.Module; forward returns the 4 tensors."""

        def __init__(self, m: GlyphMatteUNet):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self.m(x).as_tuple()

    torch.onnx.export(
        _TupleOut(model),
        (dummy,),
        out,
        input_names=["strip"],
        output_names=OUTPUT_NAMES,
        opset_version=17,
        dynamic_axes={
            "strip": {3: "width"},
            **{name: {3: "width"} for name in OUTPUT_NAMES},
        },
        do_constant_folding=True,
    )
    g = onnx.load(out)
    onnx.checker.check_model(g)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for w in (160, 320, 504):
        x = torch.randn(1, 3, HEIGHT, w)
        with torch.no_grad():
            ref = model(x)
        got = sess.run(OUTPUT_NAMES, {"strip": x.numpy()})
        for name, ref_t, got_t in zip(OUTPUT_NAMES, ref.as_tuple(), got):
            delta = float(np.abs(ref_t.numpy() - got_t).max())
            if got_t.shape[-1] != w or got_t.shape[1] != ref_t.shape[1]:
                raise RuntimeError(f"{name}: dynamic width broken at {w}")
            if delta >= 1e-3:
                raise RuntimeError(f"{name}: torch/ORT diverge at {w}: {delta}")
            print(f"width={w} {name}: max|delta|={delta:.2e}")
    print(f"onnx ok: {out} ({out.stat().st_size / 1e6:.2f} MB)")


def load_step(ckpt: Path) -> int:
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    return int(blob.get("step", 0))


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python -m versta.train.glyphmatte.export_onnx ckpt.pt out.onnx")
        sys.exit(2)
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = load_model(ckpt)
    export(model, out)


if __name__ == "__main__":
    main()
