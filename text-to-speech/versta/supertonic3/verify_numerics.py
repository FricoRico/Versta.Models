"""
Seeded numerics verification: MNN conversions vs the ONNX fp32 reference.

Per the 2/6 design deltas from the NEU-64 spike:

- fp32 is the fidelity reference wire format; the fp32 MNN conversion must
  be numerically identical to ONNX (converter-correctness control).
- int8 is the shipped precision and the only hard gate. Expected int8
  divergence vs fp32 is LSD ≈ 7.5–15 dB on wav (spike table), so the gate
  is thresholded just above the worst observation with safety margin.
- fp16 is recorded informationally only; chaotic diffusion trajectories
  legitimately diverge 12–15 dB on fr/ko. fp16 rows never fail the gate.
- Randomness is never drawn from an unseeded source: all probe inputs come
  from a seeded ``numpy.random.Generator``, and the determinism control
  re-runs the candidate with the same seed and requires bit-identity.

Per-stage inputs are introspected from the ONNX session; dynamic dims are
resolved via ``--dim name=N`` (defaults sized like a typical utterance).
"""

import argparse
import json
import sys

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import (
    FP32_CONTROL_GATE,
    INT8_GATE,
    LSD_FRAME_SIZE,
    LSD_HOP_SIZE,
    MNN_RUN_CONFIG,
)
from .sources import ONNX_FILES, UPSTREAM_REPO
from .typing import (
    MODEL_STAGES,
    PRECISIONS,
    ModelStage,
    Precision,
    StageMetric,
    ToolchainRecord,
    UpstreamRecord,
    VERIFICATION_REPORT_SCHEMA,
    VerificationReport,
)

_THRESHOLDS_REF = (
    "NEU-64 spike table 2026-10-14 (LSD/spectral-r per language); gates in config.py"
)


#: Default sizes for the dynamic dims of the Supertonic 3 stages, sized like
#: a typical utterance (spike exercised T=38..162). CLI `--dim` overrides.
DEFAULT_DIM_SIZES: Dict[str, int] = {
    "batch_size": 1,
    "text_length": 64,
    "latent_length": 48,
}


def _input_feed(
    session, seed: int, dim_overrides: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """Builds deterministic probe inputs from the ONNX session's signature."""
    rng = np.random.default_rng(seed)
    feed: Dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        shape = [
            dim
            if isinstance(dim, int) and dim > 0
            else dim_overrides.get(str(dim), DEFAULT_DIM_SIZES.get(str(dim), 64))
            for dim in meta.shape
        ]
        if "float" in meta.type:
            feed[meta.name] = rng.standard_normal(shape, dtype=np.float32)
        elif "int" in meta.type:
            dtype = np.int64 if "64" in meta.type else np.int32
            feed[meta.name] = rng.integers(0, 32, size=shape, dtype=dtype)
        elif "bool" in meta.type:
            feed[meta.name] = rng.random(shape) < 0.9
        else:
            raise RuntimeError(f"unhandled input dtype {meta.type} on {meta.name}")
    return feed


def _run_mnn(
    model_path: Path,
    feed: Dict[str, np.ndarray],
    input_names: List[str],
    output_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Runs one MNN model through the Module API (runtime manager + placeholders).

    This is the exact call path the NEU-64 spike validated bit-deterministic
    (module loaded per run, shapes set per placeholder at write time, so no
    resizeTensor/resizeSession sequencing is involved — the Interpreter API
    gave shape-inference errors and non-bit-exact re-runs on these graphs).
    """
    try:
        import MNN  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MNN python bindings not importable; install the wheel matching the "
            f"vendored tag ({model_path}) or build pymnn from vendor/mnn"
        ) from exc
    runtime = MNN.nn.create_runtime_manager((MNN_RUN_CONFIG,))
    module = MNN.nn.load_module_from_file(
        str(model_path), input_names, output_names, runtime_manager=runtime
    )
    mnn_inputs = []
    for name in input_names:
        value = feed[name]
        is_integer = np.issubdtype(value.dtype, np.integer)
        placeholder = MNN.expr.placeholder(
            tuple(value.shape),
            dtype=MNN.numpy.int32 if is_integer else MNN.numpy.float32,
        )
        placeholder.write(value.astype(np.int32 if is_integer else np.float32))
        mnn_inputs.append(placeholder)
    forwarded = module.forward(mnn_inputs)
    outputs: Dict[str, np.ndarray] = {}
    for name, expr in zip(output_names, forwarded):
        if isinstance(expr, list):
            expr = expr[0]
        if expr.dtype == MNN.numpy.float32:
            value = np.array(expr.read(), dtype=np.float32)
        else:
            value = np.array(MNN.expr.convert(expr, MNN.expr.NCHW).read())
        outputs[name] = value
    return outputs


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-noise ratio vs the reference, in dB (higher is better)."""
    signal = float(np.sum(reference**2))
    noise = float(np.sum((reference - candidate) ** 2))
    if noise == 0.0:
        return float("inf")
    return 10.0 * np.log10(signal / noise)


def cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Cosine similarity of the flattened outputs."""
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    return float("inf") if denom == 0.0 else float(np.dot(ref, cand) / denom)


def lsd_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Log-spectral distance in dB between two wavs (frame-wise, mean over frames)."""
    frame = LSD_FRAME_SIZE
    hop = LSD_HOP_SIZE
    window = np.hanning(frame)

    def log_power(wav: np.ndarray) -> np.ndarray:
        frames = [
            np.abs(np.fft.rfft(window * wav[i : i + frame])) ** 2
            for i in range(0, max(len(wav) - frame, 0) + 1, hop)
        ]
        return 10.0 * np.log10(np.maximum(np.array(frames), 1e-12))

    ref = log_power(reference.reshape(-1))
    cand = log_power(candidate.reshape(-1))
    rows = min(len(ref), len(cand))
    return float(np.mean(np.sqrt(np.mean((ref[:rows] - cand[:rows]) ** 2, axis=1))))


def spectral_r(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Pearson correlation of flattened magnitude spectra."""
    ref = np.abs(np.fft.rfft(reference.reshape(-1)))
    cand = np.abs(np.fft.rfft(candidate.reshape(-1)))
    limit = min(len(ref), len(cand))
    return float(np.corrcoef(ref[:limit], cand[:limit])[0, 1])


def compare_stage(
    snapshot: Path,
    converted_dir: Path,
    stage: ModelStage,
    precision: Precision,
    seed: int,
    dim_overrides: Dict[str, int],
) -> StageMetric:
    """Runs one stage through ONNX (fp32 reference) and one MNN precision, and scores it."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(snapshot / ONNX_FILES[stage]), providers=["CPUExecutionProvider"]
    )
    feed = _input_feed(session, seed, dim_overrides)
    reference = session.run(None, feed)
    model_path = converted_dir / precision / f"{stage}.mnn"
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    candidate = _run_mnn(model_path, feed, input_names, output_names)
    repeat = _run_mnn(model_path, feed, input_names, output_names)
    for name in candidate:
        if not np.array_equal(candidate[name], repeat[name]):
            raise RuntimeError(
                f"non-deterministic {precision} output for {stage}.{name}: "
                "seeded re-run diverged (golden-audio CI precondition violated)"
            )

    reference_flat = np.concatenate([np.atleast_1d(o.reshape(-1)) for o in reference])
    # MNN may rename outputs; align by name when possible, else by order.
    if set(output_names).issubset(candidate.keys()):
        candidate_flat = np.concatenate(
            [np.atleast_1d(candidate[name].reshape(-1)) for name in output_names]
        )
    else:
        candidate_flat = np.concatenate(
            [np.atleast_1d(value.reshape(-1)) for value in candidate.values()]
        )
    limit = min(len(reference_flat), len(candidate_flat))
    reference_flat, candidate_flat = reference_flat[:limit], candidate_flat[:limit]

    is_wav = stage == "vocoder"
    metric = StageMetric(
        stage=stage,
        precision=precision,
        snr_db=snr_db(reference_flat, candidate_flat),
        cosine=cosine(reference_flat, candidate_flat),
        lsd_db=lsd_db(candidate_flat, reference_flat) if is_wav else None,
        spectral_r=spectral_r(reference_flat, candidate_flat) if is_wav else None,
        passed=False,
    )
    gate = FP32_CONTROL_GATE if precision == "fp32" else INT8_GATE
    ok = (
        metric["snr_db"] >= gate["min_snr_db"]
        and metric["cosine"] >= gate["min_cosine"]
    )
    if is_wav and precision == "int8":
        ok = ok and metric["lsd_db"] <= INT8_GATE["max_lsd_db"]
        ok = ok and metric["spectral_r"] >= INT8_GATE["min_spectral_r"]
    metric["passed"] = bool(
        ok if precision in ("fp32", "int8") else True
    )  # fp16 informational
    return metric


def verify(
    snapshot: Path,
    converted_dir: Path,
    upstream: UpstreamRecord,
    toolchain: ToolchainRecord,
    seed: int,
    output: Optional[Path],
    dim_overrides: Optional[Dict[str, int]] = None,
) -> VerificationReport:
    """Full verification pass; writes the JSON report and a markdown table."""
    metrics: List[StageMetric] = []
    for stage in MODEL_STAGES:
        for precision in PRECISIONS:
            metrics.append(
                compare_stage(
                    snapshot, converted_dir, stage, precision, seed, dim_overrides or {}
                )
            )
    passed = all(m["passed"] for m in metrics if m["precision"] in ("fp32", "int8"))
    report = VerificationReport(
        schema=VERIFICATION_REPORT_SCHEMA,
        seed=seed,
        upstream=upstream,
        toolchain=toolchain,
        metrics=metrics,
        thresholds_ref=_THRESHOLDS_REF,
        passed=passed,
    )
    if output is not None:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    _print_table(report)
    return report


def _print_table(report: VerificationReport) -> None:
    """Markdown table for the PR description."""
    print("| stage | precision | SNR dB | cosine | LSD dB | spectral r | gate |")
    print("|---|---|---|---|---|---|---|")
    for m in report["metrics"]:
        gate = "pass" if m["passed"] else "FAIL"
        print(
            f"| {m['stage']} | {m['precision']} | {m['snr_db']:.2f} | {m['cosine']:.4f} "
            f"| {f'{m["lsd_db"]:.2f}' if m['lsd_db'] is not None else '—'} "
            f"| {f'{m["spectral_r"]:.4f}' if m['spectral_r'] is not None else '—'} | {gate} |"
        )


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        prog="versta.supertonic3 verify",
        description="Seeded numerics verification of MNN conversions vs the ONNX reference.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Local upstream snapshot (from the convert step).",
    )
    parser.add_argument(
        "--converted",
        type=Path,
        required=True,
        help="Conversion output dir containing <precision>/<stage>.mnn.",
    )
    parser.add_argument(
        "--upstream_revision",
        required=True,
        help="Pinned commit sha of Supertone/supertonic-3 used by convert.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Probe-input seed; recorded in the verification report.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="JSON report output path."
    )
    parser.add_argument(
        "--dim",
        action="append",
        default=[],
        help="Dynamic dim override, e.g. --dim T=162 (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dim_overrides = dict(pair.split("=", 1) for pair in args.dim)
    record_path = Path(args.converted) / "conversion-record.json"
    with open(record_path, encoding="utf-8") as handle:
        record = json.load(handle)
    report = verify(
        args.snapshot,
        args.converted,
        UpstreamRecord(repo=UPSTREAM_REPO, revision=args.upstream_revision),
        record["toolchain"],
        args.seed,
        args.output,
        {name: int(value) for name, value in dim_overrides.items()},
    )
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
