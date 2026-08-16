import os

from argparse import Namespace, ArgumentParser
from pathlib import Path

from .train import train_model


def parse_args() -> Namespace:
    parser = ArgumentParser(
        os.path.basename(__file__).replace(".py", ""),
        description="""Train the glyph-matte U-Net on synthetic labelled strips.""",
    )
    parser.add_argument("--size", default="16x4", help="base x levels, e.g. 16x4")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--val", type=int, default=200, help="batches per val round")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--reuse",
        type=int,
        default=16,
        help="steps each burst batch is reused on GPU",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--degrade-ramp", type=int, default=500)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile (needs a C compiler for triton)",
    )
    parser.add_argument("--weight-head", default="1x1", choices=["1x1", "3x3"])
    parser.add_argument(
        "--wrapper-width",
        type=int,
        default=384,
        help="all batches padded to this width for dynamic-shape friendliness",
    )
    return parser.parse_args()


def main() -> None:
    train_model(parse_args())


if __name__ == "__main__":
    main()
