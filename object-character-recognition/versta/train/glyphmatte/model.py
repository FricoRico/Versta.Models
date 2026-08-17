"""GlyphMatteUNet: per-strip glyph mattes for 48-px dewarped text lines.

Fully-convolutional mini U-Net: RGB strip in, four per-pixel maps out:

  matte      [1,1,H,W] — soft alpha of "is this pixel ink" (logits)
  weight     [1,1,H,W] — continuous stroke-width target 0..1 (logits)
  foreground [1,3,H,W] — decontaminated ink RGB (logits; sigmoid → 0..1)
  background [1,3,H,W] — paper RGB under the ink (logits; sigmoid → 0..1)

Ported (approach, not files) from the MIT-licensed reference implementation by
David Ventura: translator-rs/scripts/ink_model/model.py.

  https://github.com/DavidVentura/translator-rs/blob/master/scripts/ink_model/model.py

`levels=N` needs (H, W) divisible by 2**N (so levels<=4 at H=48: 48/16=3).
Each extra level enlarges the receptive field so the interiors of thick
strokes get filled, not just outlined. Layer names are kept fixed per level
so checkpoints stay compatible across level counts.
"""

from typing import NamedTuple

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Outputs(NamedTuple):
    """Ordered outputs; exported to ONNX under these exact names."""

    matte: torch.Tensor
    weight: torch.Tensor
    foreground: torch.Tensor
    background: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (self.matte, self.weight, self.foreground, self.background)


class GlyphMatteUNet(nn.Module):
    """Fully-convolutional U-Net with four per-pixel output heads.

    Encoder blocks = 2x(Conv3x3+BN+ReLU) + MaxPool down; ConvTranspose up with
    skip concatenation. All heads read the full-resolution decoder stage (d1):
    matte/weight via 1x1 (or a dilated 3x3 stack), foreground/background via a
    shared 1x1. The strip is a dewarped single line, so global colour context
    hardly needs more than d1.

    `weight_head`: "1x1" (shipped) or "3x3" (dilated stack, reference variant).
    """

    def __init__(
        self,
        base: int = 16,
        levels: int = 4,
        weight_head: str = "1x1",
    ):
        super().__init__()
        self.levels = levels
        self.weight_head_kind = weight_head

        self.enc1 = conv_block(3, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        if levels >= 3:
            self.enc4 = conv_block(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = conv_block(base * 8, base * 4)
        if levels >= 4:
            self.enc5 = conv_block(base * 8, base * 16)
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = conv_block(base * 16, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.matte_head = nn.Conv2d(base, 1, 1)
        if weight_head == "1x1":
            self.weight_head = nn.Conv2d(base, 1, 1)
        else:
            self.weight_head = nn.Sequential(
                nn.Conv2d(base, base, 3, padding=4, dilation=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base, base, 3, padding=8, dilation=8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base, 1, 1),
            )
        # foreground (ink RGB) + background (paper RGB) in one 1x1 read of d1.
        self.color_head = nn.Conv2d(base, 6, 1)

    def forward(self, x: torch.Tensor) -> Outputs:
        # Input width must be a multiple of 2**levels (pooling truncates odd
        # sizes); callers pad, training wrappers pad to wrapper-width.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        if self.levels >= 3:
            e4 = self.enc4(self.pool(e3))
            if self.levels >= 4:
                e5 = self.enc5(self.pool(e4))
                e4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
            e3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        color = self.color_head(d1)
        return Outputs(
            matte=self.matte_head(d1),
            weight=self.weight_head(d1),
            foreground=color[:, 0:3],
            background=color[:, 3:6],
        )


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class CompatUNet(GlyphMatteUNet):
    """Compatibility loader: accepts the pre-color 2-head checkpoints whose
    weight head was named `bold_head` (state key remap)."""

    @staticmethod
    def remap_state_dict(sd: dict) -> dict:
        out = {}
        for k, v in sd.items():
            out[k.replace("bold_head", "weight_head")] = v
        return out


if __name__ == "__main__":
    for levels in (2, 3, 4):
        m = GlyphMatteUNet(levels=levels)
        outs = m(torch.zeros(1, 3, 48, 320))
        shapes = {k: tuple(v.shape) for k, v in zip(outs._fields, outs.as_tuple())}
        print(f"levels={levels}: params={param_count(m):,} out={shapes}")
