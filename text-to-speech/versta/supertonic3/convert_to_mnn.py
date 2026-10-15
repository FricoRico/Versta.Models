"""
Convert the upstream Supertonic 3 ONNX stages to MNN.

Re-implementation of the community port's conversion step (`vra/
supertonic-mnn`) under our control, per the NEU-55 plan: driven by the
vendored MNN submodule (tag 3.6.1, validated by the NEU-64 spike), pinned
upstream inputs, sha256-verified passthroughs, and a checked-in ledger
record per run.

MNNConvert flags mirror what the spike validated: ``--fp16`` for the fp16
reference set and ``--weightQuantBits 8`` for the shipped int8 set
(weight-only int8; no calibration data required).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Dict, List, Optional

from .sources import (
    EXPECTED_SHA256,
    LICENSE_UPSTREAM_PATH,
    ONNX_FILES,
    PASSTHROUGH_FILES,
    UPSTREAM_REPO,
    VOICE_STYLE_FILES,
)
from .typing import (
    MODEL_STAGES,
    PRECISIONS,
    ArtifactRecord,
    ModelStage,
    Precision,
    ToolchainRecord,
    UpstreamRecord,
)

MODULE_ROOT = Path(__file__).resolve().parents[2]
MNN_SOURCE_DIR = MODULE_ROOT / "vendor" / "mnn"
MNN_TAG = "3.6.1"
MNNCONVERT_BINARY = MNN_SOURCE_DIR / "build-convert" / "MNNConvert"

#: Files downloaded from the upstream repo per run.
ALL_UPSTREAM_FILES: Dict[str, str] = {
    **ONNX_FILES,
    **PASSTHROUGH_FILES,
    **VOICE_STYLE_FILES,
    "__license__": LICENSE_UPSTREAM_PATH,
}


class PinVerificationError(RuntimeError):
    """An upstream file does not match its recorded sha256."""


def sha256_file(path: Path) -> str:
    """Streams the sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_mnnconvert(jobs: int = 8) -> Path:
    """
    Builds MNNConvert from the vendored MNN submodule (converter tools ON —
    the inverse of the app's CPU-only runtime flag set, which is deliberate:
    this pipeline is the only consumer of the converter).

    Raises:
        FileNotFoundError: If the submodule is not checked out.
        RuntimeError: If the build does not produce the binary.
    """
    if not (MNN_SOURCE_DIR / "CMakeLists.txt").exists():
        raise FileNotFoundError(
            "MNN submodule not checked out; run "
            "`git submodule update --init text-to-speech/vendor/mnn`"
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
    """Override → vendored build → PATH, else build from the submodule."""
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"MNNConvert not found at {override}")
        return override
    if MNNCONVERT_BINARY.exists():
        return MNNCONVERT_BINARY
    on_path = which("MNNConvert")
    if on_path is not None:
        return Path(on_path)
    print("MNNConvert not found; building from vendor/mnn (first run takes a while)")
    return build_mnnconvert()


def mnn_commit() -> str:
    """Commit sha of the vendored MNN submodule (recorded into the toolchain)."""
    result = subprocess.run(
        ["git", "-C", str(MNN_SOURCE_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def download_upstream(revision: str, cache_dir: Path) -> Path:
    """
    Downloads the pinned upstream snapshot and verifies every passthrough,
    style file and the LICENSE against EXPECTED_SHA256.

    Args:
        revision (str): Required upstream commit sha — moving refs are
            rejected by the ledger contract.
        cache_dir (Path): Download/staging cache.

    Returns:
        Path: Local directory of the snapshot.

    Raises:
        PinVerificationError: On any sha256 mismatch (fails closed).
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=UPSTREAM_REPO,
            revision=revision,
            allow_patterns=list(ALL_UPSTREAM_FILES.values()),
            cache_dir=str(cache_dir),
        )
    )
    for upstream_path, expected in EXPECTED_SHA256.items():
        local = snapshot / upstream_path
        if not local.exists():
            raise PinVerificationError(f"missing upstream file {upstream_path}")
        actual = sha256_file(local)
        if actual != expected:
            raise PinVerificationError(
                f"{upstream_path}: sha256 {actual} != pinned {expected}; "
                "upstream content moved — re-validate before bumping the pin"
            )
    for stage in MODEL_STAGES:
        onnx = snapshot / ONNX_FILES[stage]
        if not onnx.exists():
            raise PinVerificationError(f"missing upstream ONNX {ONNX_FILES[stage]}")
    return snapshot


def _converter_env(mnnconvert: Path) -> Dict[str, str]:
    """
    Process environment for the vendored MNNConvert: it links against shared
    libraries built next to it (and in tool/express subdirs), so
    LD_LIBRARY_PATH must cover those directories.
    """
    env = dict(os.environ)
    lib_dirs = [str(mnnconvert.parent)]
    for subdir in ("express", "tools"):
        candidate = mnnconvert.parent / subdir
        if candidate.exists():
            lib_dirs.append(str(candidate))
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def convert_stage(
    mnnconvert: Path,
    snapshot: Path,
    stage: ModelStage,
    precision: Precision,
    output_dir: Path,
) -> ArtifactRecord:
    """
    Runs MNNConvert for one stage at one precision.

    fp16 halves weights (reference set); int8 weight-quantizes (shipped);
    fp32 is the plain conversion (fidelity reference wire format).
    """
    args = [
        str(mnnconvert),
        "-f",
        "ONNX",
        "--modelFile",
        str(snapshot / ONNX_FILES[stage]),
        "--MNNModel",
        str(output_dir / precision / f"{stage}.mnn"),
    ]
    if precision == "fp16":
        args.append("--fp16")
    elif precision == "int8":
        args += ["--weightQuantBits", "8"]
    subprocess.run(args, check=True, env=_converter_env(mnnconvert))
    out = output_dir / precision / f"{stage}.mnn"
    if not out.exists():
        raise RuntimeError(f"MNNConvert did not produce {out}")
    return ArtifactRecord(
        path=str(out.relative_to(output_dir)),
        sha256=sha256_file(out),
        bytes=out.stat().st_size,
    )


def convert_all(
    revision: str, output_dir: Path, cache_dir: Path, mnnconvert: Optional[Path]
) -> Dict:
    """
    Full conversion: pinned download + verify, then all stages × precisions.

    Returns the conversion record that the bundler and ledger consume.
    """
    binary = resolve_mnnconvert(mnnconvert)
    snapshot = download_upstream(revision, cache_dir)
    upstream = UpstreamRecord(repo=UPSTREAM_REPO, revision=revision)
    toolchain = ToolchainRecord(
        mnn_commit=mnn_commit(), mnn_tag=MNN_TAG, python=platform.python_version()
    )

    artifacts: List[ArtifactRecord] = []
    for stage in MODEL_STAGES:
        for precision in PRECISIONS:
            (output_dir / precision).mkdir(parents=True, exist_ok=True)
            artifacts.append(
                convert_stage(binary, snapshot, stage, precision, output_dir)
            )
    for upstream_path, bundle_path in {
        **PASSTHROUGH_FILES,
        **VOICE_STYLE_FILES,
        "__license__": LICENSE_UPSTREAM_PATH,
    }.items():
        source = (
            snapshot / LICENSE_UPSTREAM_PATH
            if upstream_path == "__license__"
            else snapshot / upstream_path
        )
        target_name = "LICENSE" if upstream_path == "__license__" else bundle_path
        destination = output_dir / "passthrough" / target_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        artifacts.append(
            ArtifactRecord(
                path=str(destination.relative_to(output_dir)),
                sha256=sha256_file(destination),
                bytes=destination.stat().st_size,
            )
        )

    record = {
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "upstream": upstream,
        "toolchain": toolchain,
        "artifacts": artifacts,
    }
    record_path = output_dir / "conversion-record.json"
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"conversion record: {record_path}")
    return record


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        prog="versta.supertonic3 convert",
        description="Convert Supertonic 3 upstream ONNX stages to MNN (fp32/fp16/int8).",
    )
    parser.add_argument(
        "--upstream_revision",
        required=True,
        help="Pinned commit sha of Supertone/supertonic-3 (moving refs are rejected).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/supertonic3"),
        help="Conversion output directory; per-precision subdirs are created.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(".cache/supertonic3"),
        help="Hugging Face snapshot cache directory.",
    )
    parser.add_argument(
        "--mnnconvert",
        type=Path,
        default=None,
        help="Explicit MNNConvert binary; defaults to the vendored build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_all(
        args.upstream_revision, args.output_dir, args.cache_dir, args.mnnconvert
    )


if __name__ == "__main__":
    main()
