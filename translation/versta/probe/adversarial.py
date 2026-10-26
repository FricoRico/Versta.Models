"""
Adversarial control pack for the acceptance harness (NEU-60 Build 3).

Builds a deliberately **direction-mislabeled** bundle — the NEU-19 defect
class (catalog carries a model whose declared direction does not match its
payload). Construction reuses the production path end to end: the payload is
a real downloaded direction, the tarball is produced by ``versta.bundle``
itself, and the result passes every shape check the pipeline has today
(schema, hashes, leanmt binary handshake). The only thing that can catch it
is the functional probe: translating the claimed source language through a
payload that actually expects the *other* direction collapses chrF below the
corpus floor, so the direction check trips and the harness must score the
pack as ``acceptance: fail`` with the caller expectation
``direction-fail``.

Usage::

    uv run python -m versta.probe.adversarial \
        --model_dir output/en-es \
        --output_dir output/adversarial

The produced single-direction bundle claims the *reverse* of the payload's
true direction by default (en→es payload shipped as es→en), the minimal
failure a direction-sanity gate exists to catch.
"""

import argparse
import json
import shutil

from pathlib import Path
from typing import List

from ..bundle.__main__ import main as bundle_main


def build_mislabeled_bundle(
    model_dir: Path,
    output_dir: Path,
    claimed_source: str,
    claimed_target: str,
    work_dir: Path,
) -> Path:
    """
    Repacks one downloaded direction as a single-direction bundle whose
    metadata claims a different direction than the payload implements.

    Args:
        model_dir (Path): a ``versta.download`` output directory holding the
            real payload and its metadata.json.
        output_dir (Path): where the bundle tarball is written.
        claimed_source (str): language code the bundle will claim as source.
        claimed_target (str): language code the bundle will claim as target.
        work_dir (Path): scratch space (removed beforehand, kept after).

    Returns:
        Path: the produced bundle tarball.
    """
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as handle:
        real_metadata = json.load(handle)

    true_pair = f"{real_metadata['source_language']}-{real_metadata['target_language']}"
    claimed_pair = f"{claimed_source}-{claimed_target}"
    if true_pair == claimed_pair:
        raise ValueError(
            f"claimed direction {claimed_pair} equals the payload's true "
            "direction; that would not be an adversarial pack"
        )

    if work_dir.exists():
        shutil.rmtree(work_dir)
    staging = work_dir / claimed_pair
    shutil.copytree(
        model_dir,
        staging,
        ignore=shutil.ignore_patterns("metadata.json"),
    )

    mislabeled = dict(real_metadata)
    mislabeled["source_language"] = claimed_source
    mislabeled["target_language"] = claimed_target
    (staging / "metadata.json").write_text(
        json.dumps(mislabeled, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = bundle_main(
        input_dirs=[staging],
        output_dir=output_dir,
        bidirectional=False,
        keep_intermediates=False,
        keep_input=True,
    )

    # versta.bundle names the archive for the joined languages (en-es), which
    # would collide with the real pack in listings. Rename by the CLAIMED
    # direction so the defect is legible from the filename; rewrite the
    # checksum sidecar to match.
    bundle = result["bundle"]
    relabeled = bundle.with_name(f"{claimed_pair}-mislabeled-bundle.tar.gz")
    bundle.rename(relabeled)
    digest = result["checksum"].read_text(encoding="utf-8").split()[0]
    result["checksum"].unlink()
    Path(str(relabeled.with_suffix("")) + ".sha256").write_text(
        f"{digest}  {relabeled.name}\n", encoding="utf-8"
    )
    return relabeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Downloaded direction dir (versta.download output) providing the real payload.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/adversarial"),
        help="Directory for the produced mislabeled bundle tarball.",
    )
    parser.add_argument(
        "--claimed_source",
        type=str,
        default=None,
        help="Language code the bundle claims as source; defaults to the payload's true target.",
    )
    parser.add_argument(
        "--claimed_target",
        type=str,
        default=None,
        help="Language code the bundle claims as target; defaults to the payload's true source.",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("output/adversarial-work"),
        help="Scratch directory used for the relabeled staging copy.",
    )
    return parser.parse_args()


def main(
    model_dir: Path,
    output_dir: Path,
    claimed_source: str,
    claimed_target: str,
    work_dir: Path,
) -> Path:
    """
    Importable entry point: builds the mislabeled bundle and returns its path.
    """
    return build_mislabeled_bundle(
        model_dir=model_dir,
        output_dir=output_dir,
        claimed_source=claimed_source,
        claimed_target=claimed_target,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    with open(args.model_dir / "metadata.json", "r", encoding="utf-8") as handle:
        real_metadata = json.load(handle)
    bundle = main(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        claimed_source=args.claimed_source or real_metadata["target_language"],
        claimed_target=args.claimed_target or real_metadata["source_language"],
        work_dir=args.work_dir,
    )
    print(f"built mislabeled bundle: {bundle}")


__all__: List[str] = ["build_mislabeled_bundle", "main"]
