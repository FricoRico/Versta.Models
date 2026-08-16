import subprocess

from pathlib import Path
from shutil import which

from .typing import ModelSpec


def convert_to_onnx(spec: ModelSpec, model_dir: Path, onnx_path: Path) -> Path:
    """
    Converts an extracted Paddle inference model to ONNX via paddle2onnx.

    PIR-format models (Paddle 3.x, `inference.json` + `inference.pdiparams`)
    need paddle2onnx>=2.1; legacy pdmodel exports use the lower opset defined
    in the spec.

    Args:
        spec (ModelSpec): The model catalog entry.
        model_dir (Path): Directory holding the extracted inference model.
        onnx_path (Path): Destination ONNX file path.

    Returns:
        Path: The written ONNX file path.

    Raises:
        FileNotFoundError: If the paddle2onnx CLI is not on PATH.
        RuntimeError: If the conversion fails.
    """
    paddle2onnx = which("paddle2onnx")
    if paddle2onnx is None:
        raise FileNotFoundError(
            "paddle2onnx not found on PATH; run this tool via `uv run`"
        )
    model_filename = "inference.json" if spec["pir"] else "inference.pdmodel"
    result = subprocess.run(
        [
            paddle2onnx,
            "--model_dir",
            str(model_dir),
            "--model_filename",
            model_filename,
            "--params_filename",
            "inference.pdiparams",
            "--save_file",
            str(onnx_path),
            "--opset_version",
            str(spec["opset"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"paddle2onnx failed for {spec['stem']}:\n{result.stderr or result.stdout}"
        )
    return onnx_path
