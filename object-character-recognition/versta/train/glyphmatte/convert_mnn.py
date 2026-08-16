"""int8 MNN conversion for the glyphmatte ONNX, plus pack-manifest entry.

Spawns the MNNConvert binary already built for the OCR module
(`object-character-recognition/vendor/mnn/build-convert/`), so both ship the
same op support. Output `glyphmatte_int8.mnn` has four named outputs: matte,
weight, foreground, background.

CLI: `uv run python -m versta.train.glyphmatte.convert_mnn model.onnx [--out_dir output/paddle-ocr-v6]`
"""

import argparse
import hashlib
import json
import os
import subprocess

from pathlib import Path
from typing import Any, Dict, List

from .assets import MODULE_ROOT

MNN_CONVERT = MODULE_ROOT / "vendor" / "mnn" / "build-convert" / "MNNConvert"
MNN_LIBS = MNN_CONVERT.parent / "express"
MNN_CONVERT_LIBS = MNN_CONVERT.parent / "tools" / "converter"

MODEL_NAME = "glyphmatte_int8.mnn"
PRIORITY = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_mnn_convert(onnx_path: Path, out_path: Path) -> None:
    """Converts ONNX to quantized int8 MNN.

    Args:
        onnx_path (Path): Input .onnx from `export_onnx`.
        out_path (Path): Target .mnn path.

    Raises:
        RuntimeError: On failure (or success-missing-descriptor mixes).
    """
    if not MNN_CONVERT.exists():
        raise RuntimeError(
            f"MNNConvert not built at {MNN_CONVERT}; run the OCR module setup first"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    lib = ":".join([str(MNN_CONVERT.parent), str(MNN_LIBS), str(MNN_CONVERT_LIBS)])
    env["LD_LIBRARY_PATH"] = (
        lib + ":" + str(MNN_CONVERT_LIBS) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    cmd = [
        str(MNN_CONVERT),
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(out_path),
        "--bizCode",
        "mnn",
        "--weightQuantBits",
        "8",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    # MNNConvert is quirky: prints errors but sometimes still writes the file.
    if not out_path.exists():
        raise RuntimeError(
            f"MNNConvert produced no file:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"MNNConvert exit {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print(f"mnn ok: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def update_manifest(out_dir: Path, mnn_path: Path) -> None:
    """Adds/merges a `glyphmatte` entry into the target pack manifest.json
    (default the OCR pack) so the produced model ships with the rest.

    Args:
        out_dir (Path): Pack directory containing manifest.json.
        mnn_path (Path): The produced .mnn file inside `out_dir`.
    """
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest: Dict[str, Any] = json.loads(manifest_path.read_text())
    else:
        version = (MODULE_ROOT / "versta" / "version.txt").read_text().strip()
        manifest = {"pack": "paddle-ocr-v6", "version": version, "files": []}
    files: List[Dict[str, Any]] = [
        f for f in manifest.get("files", []) if f.get("name") != MODEL_NAME
    ]
    files.append(
        {
            "name": MODEL_NAME,
            "sizeBytes": mnn_path.stat().st_size,
            "sha256": sha256_file(mnn_path),
            "role": "glyphmatte",
            "priority": PRIORITY,
            "note": "per-line glyph matte: outputs matte(1), weight(1)=stroke-width, foreground(3)=ink RGB, background(3)=paper RGB",
        }
    )
    files.sort(
        key=lambda f: (f.get("priority", 0), f.get("role", ""), f.get("name", ""))
    )
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest updated: {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx", type=Path)
    p.add_argument("--out_dir", type=Path, default=Path("output/paddle-ocr-v6"))
    p.add_argument("--no_manifest", action="store_true")
    args = p.parse_args()
    mnn_path = args.out_dir / MODEL_NAME
    run_mnn_convert(args.onnx, mnn_path)
    if not args.no_manifest:
        update_manifest(args.out_dir, mnn_path)


if __name__ == "__main__":
    main()
