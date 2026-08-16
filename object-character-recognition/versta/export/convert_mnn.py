import os
import subprocess

from pathlib import Path
from shutil import which
from typing import Dict, Optional

MODULE_ROOT = Path(__file__).resolve().parents[2]
MNN_SOURCE_DIR = MODULE_ROOT / "vendor" / "MNN"
MNNCONVERT_BINARY = MNN_SOURCE_DIR / "build-convert" / "MNNConvert"


def build_mnnconvert(jobs: int = 8) -> Path:
    """
    Builds MNNConvert from the vendored MNN source tree (submodule pinned to
    the MNN release tag the pack targets).

    Args:
        jobs (int): Parallel build jobs.

    Returns:
        Path: The built MNNConvert binary path.

    Raises:
        FileNotFoundError: If the MNN submodule is not checked out.
        RuntimeError: If the build fails or does not produce the binary.
    """
    if not (MNN_SOURCE_DIR / "CMakeLists.txt").exists():
        raise FileNotFoundError(
            "MNN submodule not checked out; run "
            "`git submodule update --init object-character-recognition/vendor/MNN`"
        )
    build_dir = MNN_SOURCE_DIR / "build-convert"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(MNN_SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DMNN_BUILD_CONVERTER=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "MNNConvert", "-j", str(jobs)],
        check=True,
    )
    if not MNNCONVERT_BINARY.exists():
        raise RuntimeError(f"MNNConvert build did not produce {MNNCONVERT_BINARY}")
    return MNNCONVERT_BINARY


def resolve_mnnconvert(override: Optional[Path]) -> Path:
    """
    Resolves the MNNConvert binary: an explicit override, a binary already on
    PATH, or the vendored build — building it from the submodule on first use.

    Args:
        override (Optional[Path]): Explicit MNNConvert path, or None.

    Returns:
        Path: The MNNConvert binary path.

    Raises:
        FileNotFoundError: If no binary can be resolved or built.
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"MNNConvert not found at {override}")
        return override
    if MNNCONVERT_BINARY.exists():
        return MNNCONVERT_BINARY
    on_path = which("MNNConvert")
    if on_path is not None:
        return Path(on_path)
    print("MNNConvert not found; building from vendor/MNN (first run takes a while)")
    return build_mnnconvert()


def _converter_env(mnnconvert: Path) -> Dict[str, str]:
    """
    Builds the process environment for the vendored MNNConvert: it links
    against shared libraries built next to it (and in tool/express subdirs),
    so LD_LIBRARY_PATH must cover those directories.

    Args:
        mnnconvert (Path): The MNNConvert binary.

    Returns:
        Dict[str, str]: The environment for the converter subprocess.
    """
    env = dict(os.environ)
    dirs = [
        mnnconvert.parent,
        mnnconvert.parent / "tools" / "converter",
        mnnconvert.parent / "express",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in dirs if d.exists()) + (
        f":{existing}" if existing else ""
    )
    return env


def convert_to_mnn(mnnconvert: Path, onnx_path: Path, mnn_path: Path) -> Path:
    """
    Converts an ONNX model to MNN with int8 weight quantization
    (`--weightQuantBits 8`).

    Args:
        mnnconvert (Path): The MNNConvert binary.
        onnx_path (Path): Source ONNX model.
        mnn_path (Path): Destination MNN model.

    Returns:
        Path: The written MNN model path.

    Raises:
        RuntimeError: If the conversion fails.
    """
    mnn_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(mnnconvert),
            "-f",
            "ONNX",
            "--modelFile",
            str(onnx_path),
            "--MNNModel",
            str(mnn_path),
            "--bizCode",
            "biz",
            "--weightQuantBits",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_converter_env(mnnconvert),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MNNConvert failed for {onnx_path.name}:\n{result.stderr or result.stdout}"
        )
    return mnn_path
