"""Evaluation: synthetic-val metrics, ONNX↔int8 quantization drift, on-device figures.

Reports, on `--n` freshly generated validation strips (degrade ON, fixed seed):
  - matte IoU at 0.5
  - weight MAE on ink pixels and a weight-ordering AUC
  - foreground MAE (ink pixels) and background MAE (paper pixels) after sigmoid
  - ONNX↔int8 drift (symmetric matte IoU + per-output |Δ| between the graphs)
  - file size + CPU wall-clock per strip through MNN (`execution=[CPU]`),
    the number the app actually pays for.

Predictions are dicts keyed by output name: matte / weight / foreground /
background. torch and MNN go through a lazy `_mnn()` import that auto-applies
the PyMNN execstack patch (see `fix_mnn_execstack`).

CLI: uv run python -m versta.train.glyphmatte.eval --mnn output/glyphmatte/glyphmatte_int8.mnn --onnx output/glyphmatte.onnx
"""

import argparse
import json
import sysconfig
import time

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...dataset.glyphmatte.parquet import read_shards
from .config import LAYOUT, OUTPUT_NAMES, TRAIN

ValRow = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Pred = Dict[str, np.ndarray]


def _mnn():
    """Imports PyMNN, patching the execstack-marked engine library first if
    the system loader rejects it."""
    try:
        import MNN
    except ImportError:
        from .fix_mnn_execstack import patch

        n = patch(Path(sysconfig.get_paths()["purelib"]))
        print(f"patched {n} execstack segment(s); retrying MNN import")
        import MNN
    return MNN


def _val_from_shards(shards, n: int) -> List[ValRow]:
    out: List[ValRow] = []
    for strip in read_shards(shards):
        out.append(
            (
                strip.rgb.transpose(2, 0, 1).astype(np.float32),
                strip.matte,
                strip.weight,
                strip.fcol,
                strip.bcol,
            )
        )
        if len(out) >= n:
            break
    return out


def pad_to_multiple(img: np.ndarray, m: int = 16) -> np.ndarray:
    """Zero-pads width up to a multiple of `m` (2**levels of the U-Net);
    outputs must be cropped back by the caller (callers do so via crop_to)."""
    w = img.shape[2]
    pad = (-w) % m
    if not pad:
        return img
    return np.pad(img, ((0, 0), (0, 0), (0, pad)))


def crop_w(arr: np.ndarray, w: int) -> np.ndarray:
    return arr[..., :w]


def _predict_mnn(model_path: Path, imgs: List[np.ndarray]) -> List[Pred]:
    """Per-strip MNN inference (one session per call, CPU); outputs by name."""
    MNN = _mnn()

    interpreter = MNN.Interpreter(str(model_path))
    session = interpreter.createSession({"thread": 4, "backend": "CPU"})
    input_tensor = interpreter.getSessionInput(session)
    outs: List[Pred] = []
    for img in imgs:
        w_orig = img.shape[2]
        img = pad_to_multiple(img)
        h, w = img.shape[1], img.shape[2]
        interpreter.resizeTensor(input_tensor, (1, 3, h, w))
        interpreter.resizeSession(session)
        input_tensor = interpreter.getSessionInput(session)
        tmp = MNN.Tensor(
            (1, 3, h, w), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe
        )
        input_tensor.copyFrom(tmp)
        interpreter.runSession(session)
        pred: Pred = {}
        for name in OUTPUT_NAMES:
            t = interpreter.getSessionOutput(session, name)
            host = MNN.Tensor(
                t.getShape(),
                MNN.Halide_Type_Float,
                t.getData(),
                MNN.Tensor_DimensionType_Caffe,
            )
            t.copyToHostTensor(host)
            arr = np.array(host.getData(), dtype=np.float32).reshape(host.getShape())[0]
            pred[name] = crop_w(arr, w_orig)
        outs.append(pred)
    return outs


def _predict_onnx(onnx_path: Path, imgs: List[np.ndarray]) -> List[Pred]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs: List[Pred] = []
    for img in imgs:
        w_orig = img.shape[2]
        res = sess.run(list(OUTPUT_NAMES), {"strip": pad_to_multiple(img)[None]})
        outs.append(
            {name: crop_w(res[i][0], w_orig) for i, name in enumerate(OUTPUT_NAMES)}
        )
    return outs


def _predict_torch(model: torch.nn.Module, imgs: List[np.ndarray]) -> List[Pred]:
    outs: List[Pred] = []
    with torch.no_grad():
        for img in imgs:
            w_orig = img.shape[2]
            img = pad_to_multiple(img)
            o = model(torch.from_numpy(img[None]))
            outs.append(
                {
                    k: crop_w(v.numpy()[0], w_orig)
                    for k, v in zip(o._fields, o.as_tuple(), strict=True)
                }
            )
    return outs


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def matte_iou(pred: Pred, label: np.ndarray) -> float:
    pm = _sig(pred["matte"][0]) > 0.5
    inter = float((pm & (label > 0.5)).sum())
    union = float((pm | (label > 0.5)).sum())
    return inter / max(union, 1.0)


def weight_metrics(
    pred: Pred, matte: np.ndarray, weight: np.ndarray
) -> Tuple[float, float]:
    """Weight MAE on ink pixels + ordering AUC (pairwise, sampled)."""
    ink = matte > 0.5
    if ink.sum() < 30:
        return float("nan"), float("nan")
    p = _sig(pred["weight"][0])[ink]
    t = weight[ink]
    mae = float(np.abs(p - t).mean())
    idx = np.argsort(t)
    p, t = p[idx], t[idx]
    rng = np.random.default_rng(0)
    n = len(t)
    take = min(20000, n)
    i = rng.integers(0, n, take)
    j = rng.integers(0, n, take)
    ok = np.abs(t[i] - t[j]) > 0.25
    if not ok.any():
        return mae, float("nan")
    wins = (p[i][ok] - p[j][ok]) * np.sign(t[i] - t[j])[ok] > 0
    ties = p[i][ok] == p[j][ok]
    auc = float((wins.sum() + 0.5 * ties.sum()) / ok.sum())
    return mae, auc


def color_metrics(
    pred: Pred, matte: np.ndarray, fg: np.ndarray, bg: np.ndarray
) -> Tuple[float, float]:
    """Per-channel L1 after sigmoid; foreground scored on ink, background on paper."""
    ink = matte > 0.5
    paper = matte < 0.5
    f_mae = (
        float(np.abs(_sig(pred["foreground"].transpose(1, 2, 0))[ink] - fg[ink]).mean())
        if ink.sum() >= 30
        else float("nan")
    )
    b_mae = (
        float(
            np.abs(
                _sig(pred["background"].transpose(1, 2, 0))[paper] - bg[paper]
            ).mean()
        )
        if paper.sum() >= 30
        else float("nan")
    )
    return f_mae, b_mae


def summarize(name: str, preds: List[Pred], val: List[ValRow]) -> Dict[str, float]:
    ious, w_maes, w_aucs, f_maes, b_maes = [], [], [], [], []
    for pred, (_, matte, weight, fg, bg) in zip(preds, val):
        ious.append(matte_iou(pred, matte))
        mae, auc = weight_metrics(pred, matte, weight)
        w_maes.append(mae)
        w_aucs.append(auc)
        f, b = color_metrics(pred, matte, fg, bg)
        f_maes.append(f)
        b_maes.append(b)
    out = {
        "matte_iou": float(np.nanmean(ious)),
        "matte_iou_p5": float(np.nanpercentile(ious, 5)),
        "weight_mae": float(np.nanmean(w_maes)),
        "weight_auc": float(np.nanmean(w_aucs)),
        "foreground_mae": float(np.nanmean(f_maes)),
        "background_mae": float(np.nanmean(b_maes)),
    }
    print(f"{name}: " + " ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def model_parity(a: List[Pred], b: List[Pred]) -> Dict[str, float]:
    """Agreement between two model outputs on the same strips: matte IoU plus
    mean |Δ| over the other three outputs."""
    ious, deltas = [], []
    for pa, pb in zip(a, b):
        ma = _sig(pa["matte"][0]) > 0.5
        mb = _sig(pb["matte"][0]) > 0.5
        ious.append(float((ma & mb).sum() / max((ma | mb).sum(), 1)))
        per = []
        for name in ("weight", "foreground", "background"):
            per.append(float(np.abs(_sig(pa[name]) - _sig(pb[name])).mean()))
        deltas.append(float(np.mean(per)))
    out = {
        "parity_matte_iou": float(np.mean(ious)),
        "parity_other_mae": float(np.mean(deltas)),
    }
    print("parity: " + " ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def touch_speed(mnn_path: Path, imgs: List[np.ndarray], reps: int = 3) -> float:
    """ms per strip through the MNN session (CPU, thread=4)."""
    MNN = _mnn()

    interpreter = MNN.Interpreter(str(mnn_path))
    session = interpreter.createSession({"thread": 4, "backend": "CPU"})
    ip = interpreter.getSessionInput(session)
    t0 = time.perf_counter()
    n = 0
    for _ in range(reps):
        for img in imgs:
            img = pad_to_multiple(img)
            h, w = img.shape[1], img.shape[2]
            interpreter.resizeTensor(ip, (1, 3, h, w))
            interpreter.resizeSession(session)
            ip = interpreter.getSessionInput(session)
            tmp = MNN.Tensor(
                (1, 3, h, w), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe
            )
            ip.copyFrom(tmp)
            interpreter.runSession(session)
            n += 1
    return (time.perf_counter() - t0) * 1000 / n


def evaluate(
    ckpt: Optional[Path] = None,
    onnx: Optional[Path] = None,
    mnn: Optional[Path] = None,
    n: int = TRAIN.eval_n,
    dataset_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Val metrics; returns the results dict.

    Validation source: the materialized `validation-*.parquet` shards under
    ``dataset_dir``.
    """
    shards = sorted(dataset_dir.glob("validation-*.parquet")) if dataset_dir else []
    if not shards:
        raise RuntimeError(f"no validation shards found in {dataset_dir}")
    print(f"val from {shards[0].parent}")
    val = _val_from_shards(shards, n)
    imgs = [v[0] for v in val]
    results: Dict[str, Dict[str, float]] = {}

    if ckpt:
        from .export_onnx import load_model

        results["torch_fp32"] = summarize(
            "torch", _predict_torch(load_model(ckpt), imgs), val
        )

    onnx_preds: Optional[List[Pred]] = None
    if onnx and onnx.exists():
        onnx_preds = _predict_onnx(onnx, imgs)
        results["onnx"] = summarize("onnx ", onnx_preds, val)

    if mnn and mnn.exists():
        mnn_preds = _predict_mnn(mnn, imgs)
        results["mnn_int8"] = summarize("mnn  ", mnn_preds, val)
        if onnx_preds:
            results["onnx_vs_mnn"] = model_parity(onnx_preds, mnn_preds)
        ms = touch_speed(mnn, imgs[:8])
        size_mb = mnn.stat().st_size / 1e6
        print(f"mnn size={size_mb:.2f} MB  speed={ms:.1f} ms/strip (cpu x4)")
        results["mnn_int8"]["size_mb"] = size_mb
        results["mnn_int8"]["ms_per_strip"] = ms

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="pytorch checkpoint to score (default: intermediates latest).",
    )
    parser.add_argument("--onnx", type=Path, default=None, help="fp32 ONNX to score.")
    parser.add_argument("--mnn", type=Path, default=None, help="int8 MNN to score.")
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the results JSON here."
    )
    args = parser.parse_args()
    out_dir = Path("output")
    ckpt = args.ckpt or out_dir / LAYOUT.ckpt_latest
    onnx = args.onnx or out_dir / LAYOUT.onnx
    results = evaluate(ckpt, onnx, args.mnn, dataset_dir=out_dir / LAYOUT.dataset)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
