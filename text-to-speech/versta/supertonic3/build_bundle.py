"""
Assemble and lint the shippable `supertonic3-bundle.tar.gz`.

Bundle contents (NEU-68 scope + NEU-64 OpenRAIL obligations):

- the four int8 ``.mnn`` stages (fp16/fp32 stay out — gate references only);
- the byte-identical passthroughs (``tts.json``, ``unicode_indexer.json``,
  ``voice_styles/*.json``) verified by sha256 at conversion time;
- the upstream BigScience Open RAIL-M LICENSE copy (the community port
  ships none — we must carry the text);
- a generated README with the prominent modification notice (§III.4):
  what was converted, quantization level, source revision, build date;
- ``metadata.json`` in the app-load contract shape (see metadata.py).

Lint is fail-fast: a bundle missing a required file, the LICENSE, or
exceeding the 120 MB int8 cap is never produced.
"""

import argparse
import gzip
import json
import tarfile

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

from .config import BUNDLE_SIZE_CAP_BYTES, REQUIRED_BUNDLE_FILES
from .convert_to_mnn import sha256_file
from .metadata import build_metadata
from .sources import BUNDLE_FILENAME
from .typing import MODEL_STAGES

README_TEMPLATE = """# Supertonic 3 — Versta on-device voice bundle

Derived from `{upstream_repo}` at revision `{upstream_revision}`.
Licensed under the BigScience Open RAIL-M License (see `LICENSE`); use
restrictions in Attachment A apply and are passed through to recipients.

## Modification notice (Open RAIL-M §III.4)

- Converted ONNX → MNN with MNNConvert (MNN {mnn_tag}, commit {mnn_commit}).
- Weights quantized weight-only to int8 (fp32/fp16 conversions are produced
  as quality-gate references and are not part of this bundle).
- Built {built_at} UTC by the Versta.Models supertonic3 pipeline (NEU-68).

Voice coverage: 31 languages × 2 genders = 62 voices; ten style presets
under `voice_styles/`. Engine semantics (sample rate 44100, 5 diffusion
steps) are unchanged from upstream; see `tts.json`.
"""


def assemble_bundle_dir(converted_dir: Path, staging: Path) -> Path:
    """
    Copies the int8 set + passthroughs + license + generated docs into a
    staging root shaped exactly like the app-facing "model root".

    Raises:
        FileNotFoundError: If any required conversion output is missing.
    """
    import shutil

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for stage in MODEL_STAGES:
        source = converted_dir / "int8" / f"{stage}.mnn"
        if not source.exists():
            raise FileNotFoundError(f"missing int8 stage {source} — run convert first")
        shutil.copy2(source, staging / f"{stage}.mnn")
    passthrough_root = converted_dir / "passthrough"
    for relative in REQUIRED_BUNDLE_FILES:
        if relative.endswith(".mnn") or relative in ("README.md", "metadata.json"):
            continue
        source = passthrough_root / relative
        if not source.exists():
            raise FileNotFoundError(f"missing passthrough {source} — run convert first")
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return staging


def write_readme(staging: Path, record: Dict) -> Path:
    """Writes the generated README carrying the OpenRAIL modification notice."""
    readme = staging / "README.md"
    readme.write_text(
        README_TEMPLATE.format(
            upstream_repo=record["upstream"]["repo"],
            upstream_revision=record["upstream"]["revision"],
            mnn_tag=record["toolchain"]["mnn_tag"],
            mnn_commit=record["toolchain"]["mnn_commit"],
            built_at=record["converted_at"],
        ),
        encoding="utf-8",
    )
    return readme


def lint_bundle(staging: Path) -> List[str]:
    """
    Fail-fast lint of the staged bundle. Returns the list of violations
    (empty = shippable).
    """
    violations: List[str] = []
    for relative in REQUIRED_BUNDLE_FILES:
        if not (staging / relative).exists():
            violations.append(f"missing required file: {relative}")
    metadata = build_metadata(version="__lint__")
    for section in metadata["files"]["inference"]["models"].values():
        if not (staging / section).exists():
            violations.append(f"metadata.json references missing file: {section}")
    return violations


def write_tarball(staging: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Writes the deterministic tarball + .sha256 sidecar (sorted entries,
    clamped mtimes → byte-reproducible given identical inputs).

    Raises:
        RuntimeError: If the tarball exceeds the 120 MB int8 cap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tarball = output_dir / BUNDLE_FILENAME
    entries = sorted(path for path in staging.rglob("*") if path.is_file())
    with open(tarball, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in entries:
                    info = tar.gettarinfo(
                        str(path), arcname=str(path.relative_to(staging))
                    )
                    info.mtime = 0
                    with open(path, "rb") as handle:
                        tar.addfile(info, handle)
    if tarball.stat().st_size > BUNDLE_SIZE_CAP_BYTES:
        raise RuntimeError(
            f"bundle {tarball.stat().st_size} bytes exceeds the 120 MB int8 cap"
        )
    checksum = tarball.with_suffix(".sha256")
    checksum.write_text(sha256_file(tarball), encoding="utf-8")
    return tarball, checksum


def bundle(converted_dir: Path, output_dir: Path, version: str) -> Tuple[Path, Path]:
    """Full bundle build: assemble → readme → metadata → lint → tarball+sha256."""
    with open(converted_dir / "conversion-record.json", encoding="utf-8") as handle:
        record = json.load(handle)
    staging = output_dir / "supertonic3"
    assemble_bundle_dir(converted_dir, staging)
    write_readme(staging, record)
    metadata = build_metadata(version)
    with open(staging / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    violations = lint_bundle(staging)
    if violations:
        raise RuntimeError("bundle lint failed: " + "; ".join(violations))
    tarball, checksum = write_tarball(staging, output_dir)
    print(f"bundle: {tarball} ({tarball.stat().st_size} bytes)")
    print(f"sha256: {checksum.read_text().strip()}")
    return tarball, checksum


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        prog="versta.supertonic3 bundle",
        description="Build the shippable supertonic3 bundle tarball + sha256.",
    )
    parser.add_argument(
        "--converted",
        type=Path,
        required=True,
        help="Conversion output dir (from `convert`).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Directory the tarball and .sha256 are written to.",
    )
    parser.add_argument("--version", required=True, help="Bundle version, e.g. v2.0.0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle(args.converted, args.output_dir, args.version)


if __name__ == "__main__":
    main()
