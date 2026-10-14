"""
Evaluate a produced translation direction against the pinned corpus:

    uv run python -m versta.eval --source en --target es \
        --bundle output/en-es-bundle.tar.gz

or, for a pre-bundling artifact:

    uv run python -m versta.eval --source en --target es \
        --direction_dir output/en-es

One command evaluates one direction and writes a versioned result JSON
(scores, corpus revision, engine version, artifact checksums) for the
release gate and the ``evals/`` ledger. The reference runtime is the
upstream bergamot-translator CLI until the LeanMT batch mode (NEU-50)
lands; configure it with --bergamot_bin/--bergamot_image (env defaults:
VERSTA_BERGAMOT_BIN / VERSTA_BERGAMOT_IMAGE).
"""

import os

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from .engines import create_engine
from .evaluate import DEFAULT_OUTPUT_DIR, evaluate_direction


def parse_args() -> Namespace:
    parser = ArgumentParser(
        os.path.basename(__file__),
        description="""Evaluate one produced translation direction: translate the pinned
        FLORES-200 corpus snapshot through the reference runtime and score
        sacreBLEU + chrF, recording provenance (engine version, artifact
        checksums, corpus revision) into a result JSON.""",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source language code (e.g. 'en').",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target language code (e.g. 'es').",
    )

    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument(
        "--direction_dir",
        type=Path,
        help="A versta.download output directory (pre-bundling artifact).",
    )
    artifact.add_argument(
        "--bundle",
        type=Path,
        help="A shipped *-bundle.tar.gz; the direction is extracted and "
        "evaluated from the exact artifact the app side-loads.",
    )

    parser.add_argument(
        "--corpus_manifest",
        type=Path,
        default=Path("evals") / "corpora.json",
        help="The pinned corpus manifest (default: evals/corpora.json).",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="bergamot",
        choices=["bergamot", "leanmt"],
        help="Reference runtime. 'bergamot' drives the upstream "
        "bergamot-translator CLI (v1 default); 'leanmt' is reserved for the "
        "LeanMT batch CLI (NEU-50, not landed yet).",
    )

    parser.add_argument(
        "--bergamot_bin",
        type=Path,
        default=None,
        help="Path to a local bergamot CLI binary. Env: VERSTA_BERGAMOT_BIN.",
    )

    parser.add_argument(
        "--bergamot_image",
        type=str,
        default=None,
        help="Container image whose default command is the bergamot CLI "
        "(invoked via docker run). Env: VERSTA_BERGAMOT_IMAGE. Exactly one of "
        "--bergamot_bin / --bergamot_image is required.",
    )

    parser.add_argument(
        "--comet",
        action="store_true",
        default=False,
        help="Also score COMET-22 (release tier; needs the optional "
        "unbabel-comet dependency and downloads the scoring model).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result JSON destination (default: output/evals/<src>-<tgt>.json).",
    )

    parser.add_argument(
        "--keep_workdir",
        action="store_true",
        default=False,
        help="Keep the engine work directory (config, inputs, raw hypotheses) "
        "for debugging instead of cleaning it up.",
    )

    return parser.parse_args()


def main(
    source: str,
    target: str,
    direction_dir: Optional[Path],
    bundle: Optional[Path],
    corpus_manifest: Path,
    engine: str,
    bergamot_bin: Optional[Path],
    bergamot_image: Optional[str],
    comet: bool,
    output: Optional[Path],
    keep_workdir: bool,
) -> Path:
    """
    Runs one direction eval and writes the result JSON.

    Args:
        source (str): Source language code.
        target (str): Target language code.
        direction_dir (Path | None): versta.download output directory.
        bundle (Path | None): Shipped bundle tarball (exclusive with
            direction_dir).
        corpus_manifest (Path): Pinned corpus manifest path.
        engine (str): Engine identifier ("bergamot" or reserved "leanmt").
        bergamot_bin (Path | None): Local bergamot binary.
        bergamot_image (str | None): Bergamot container image.
        comet (bool): Also score COMET-22.
        output (Path | None): Result JSON destination.
        keep_workdir (bool): Keep the engine work directory.

    Returns:
        Path: The written result JSON path.
    """
    binary = bergamot_bin or (
        Path(os.environ["VERSTA_BERGAMOT_BIN"])
        if os.environ.get("VERSTA_BERGAMOT_BIN")
        else None
    )
    image = bergamot_image or os.environ.get("VERSTA_BERGAMOT_IMAGE") or None

    runtime = create_engine(engine, bergamot_binary=binary, bergamot_image=image)

    result = evaluate_direction(
        source=source,
        target=target,
        engine=runtime,
        corpus_manifest_path=corpus_manifest,
        direction_dir=direction_dir,
        bundle=bundle,
        with_comet=comet,
        output_path=output or (DEFAULT_OUTPUT_DIR / f"{source}-{target}.json"),
        keep_workdir=keep_workdir,
    )

    print(
        f"{result['direction']}: bleu={result['scores']['bleu']} "
        f"chrf={result['scores']['chrf']}"
        + (
            f" comet22={result['scores']['comet22']}"
            if result["scores"]["comet22"] is not None
            else ""
        )
    )
    return output or (DEFAULT_OUTPUT_DIR / f"{source}-{target}.json")


if __name__ == "__main__":
    args = parse_args()
    main(
        source=args.source,
        target=args.target,
        direction_dir=args.direction_dir,
        bundle=args.bundle,
        corpus_manifest=args.corpus_manifest,
        engine=args.engine,
        bergamot_bin=args.bergamot_bin,
        bergamot_image=args.bergamot_image,
        comet=args.comet,
        output=args.output,
        keep_workdir=args.keep_workdir,
    )
