import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

from PIL import Image

with open(Path(__file__).parent / ".." / "version.txt", "r") as version_file:
    version = version_file.read().strip()

ONNX_URL = "https://offline-translator.davidv.dev/support/1/docaligner_lcnet050.onnx"
INPUT_SIZE = 256


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="""Validate the exported DocAligner MNN model against CPU onnxruntime:
        corner heatmap/point maxima must agree within a few pixels on synthetic
        document photos."""
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to the reference fp32 ONNX model. If omitted, it is downloaded"
        " into <work_dir>.",
    )
    parser.add_argument(
        "--mnn",
        type=Path,
        default=Path("output") / "paddle-ocr-v6" / "docaligner_lcnet050_int8.mnn",
        help="Path to the exported int8 MNN model.",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("output") / "docaligner-check",
        help="Scratch directory for the downloaded ONNX model.",
    )
    parser.add_argument(
        "--samples", type=int, default=16, help="Number of synthetic document photos."
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _perspective_coeffs(
    target: List[Tuple[float, float]], source: List[Tuple[float, float]]
) -> List[float]:
    """PIL PERSPECTIVE coefficients mapping output coords (target quad) to the
    source image (source quad)."""
    matrix = []
    for s, t in zip(source, target):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    a_mat = np.array(matrix, dtype=np.float64)
    b_vec = np.array(source, dtype=np.float64).reshape(8)
    return np.linalg.solve(a_mat, b_vec).tolist()


def make_photo(
    rng: np.random.Generator, size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """A synthetic document photo: a text-like page perspective-warped onto a
    busy background. Returns (image uint8 HWC, corners (4,2) in photo pixels)."""
    w, h = size
    pw, ph = int(rng.integers(300, 380)), int(rng.integers(400, 520))
    page = np.full((ph, pw, 3), 250, dtype=np.uint8)
    rows = int(rng.integers(10, 20))
    for r in range(rows):
        y = int(ph * (0.07 + 0.9 * r / rows))
        x_end = int(pw * rng.uniform(0.4, 0.92))
        page[y : y + 6, int(pw * 0.08) : x_end] = rng.integers(10, 70, size=3)
    if rng.random() < 0.5:
        page = (
            (page.astype(np.int16) + rng.normal(0, 4, page.shape).astype(np.int16))
            .clip(0, 255)
            .astype(np.uint8)
        )

    bg = (
        (
            rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8).astype(np.int16)
            + rng.normal(0, 12, (h, w, 3)).astype(np.int16)
        )
        .clip(0, 255)
        .astype(np.uint8)
    )

    # Random quadrilateral for the page, kept well inside the frame (TL TR BR BL).
    def jittered(ix: int, iy: int) -> Tuple[float, float]:
        cx = 0.5 + ix * 0.3 + rng.uniform(-0.16, 0.16)
        cy = 0.5 + iy * 0.3 + rng.uniform(-0.16, 0.16)
        return (cx * w, cy * h)

    corners = np.array(
        [jittered(-1, -1), jittered(1, -1), jittered(1, 1), jittered(-1, 1)],
        dtype=np.float64,
    )
    # PIL PERSPECTIVE uses the inverse map: photo target coords -> page source coords.
    coeffs = _perspective_coeffs(
        [(float(x), float(y)) for x, y in corners], [(0, 0), (pw, 0), (pw, ph), (0, ph)]
    )
    bg_img = Image.fromarray(bg)
    warped_page = Image.fromarray(page).transform(
        (w, h), Image.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
    )
    mask = Image.new("L", (pw, ph), 255).transform(
        (w, h), Image.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR
    )
    return np.asarray(
        Image.composite(warped_page, bg_img, mask), dtype=np.uint8
    ), corners


def _run_ort(onnx_path: Path, x: np.ndarray) -> Tuple[np.ndarray, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"img": x})
    points = np.array(outs[0], dtype=np.float32).reshape(8)
    has_obj = float(np.array(outs[1]).reshape(-1)[0])
    return points, has_obj


def _run_mnn(mnn_path: Path, x: np.ndarray) -> Tuple[np.ndarray, float]:
    import MNN

    interpreter = MNN.Interpreter(str(mnn_path))
    session = interpreter.createSession({"numThread": 1, "precision": 1})
    input_tensor = interpreter.getSessionInput(session, "img")
    tmp = MNN.Tensor(
        (1, 3, INPUT_SIZE, INPUT_SIZE),
        MNN.Halide_Type_Float,
        np.ascontiguousarray(x, dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    input_tensor.copyFrom(tmp)
    interpreter.runSession(session)

    def read(name: str) -> np.ndarray:
        out = interpreter.getSessionOutput(session, name)
        shape = out.getShape()
        host = MNN.Tensor(
            tuple(int(d) for d in shape),
            MNN.Halide_Type_Float,
            np.zeros(tuple(int(d) for d in shape), dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        out.copyToHostTensor(host)
        try:
            data = host.getNumpyData()
        except AttributeError:
            data = np.array(host.getData(), dtype=np.float32)
        return np.array(data, dtype=np.float32).reshape(shape)

    points = read("points")
    return points.reshape(8), float(read("has_obj").reshape(-1)[0])


def preprocess(img: np.ndarray) -> np.ndarray:
    resized = Image.fromarray(img).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def main() -> None:
    args = parse_args()

    work_dir: Path = args.work_dir
    onnx_path = args.onnx
    if onnx_path is None:
        from .download import download_file

        onnx_path = download_file(ONNX_URL, work_dir / "docaligner_lcnet050.onnx")

    rng = np.random.default_rng(args.seed)
    diffs: List[float] = []
    worst_gt = 0.0
    worst_obj = 0.0
    for i in range(args.samples):
        img, gt = make_photo(rng, (480 + int(rng.integers(0, 160)), 480))
        x = preprocess(img)
        p_ort, o_ort = _run_ort(onnx_path, x)
        p_mnn, o_mnn = _run_mnn(args.mnn, x)
        obj_diff = abs(o_ort - o_mnn)

        # Points are normalized 0..1 over the (256x256) input.
        w, h = img.shape[1], img.shape[0]
        back_diff_px = float(np.abs(p_ort - p_mnn).max()) * INPUT_SIZE
        pred_pix = p_ort.reshape(4, 2) * np.array([w, h], dtype=np.float64)
        # Match predicted corners to the nearest GT corner (order-agnostic).
        dists = np.linalg.norm(pred_pix[:, None] - gt[None], axis=2)
        gt_err = float(dists.min(axis=1).max())
        diffs.append(back_diff_px)
        worst_obj = max(worst_obj, obj_diff)
        worst_gt = max(worst_gt, gt_err)
        print(
            f"sample {i:02d}: |Δback|={back_diff_px:.2f}in-px |Δobj|={obj_diff:.4f} "
            f"gt_err={gt_err:.1f}px (has_obj={o_ort:.2f})"
        )

    arr = np.array(diffs)
    print(
        f"|Δback| in-px: mean={arr.mean():.2f} p95={np.percentile(arr, 95):.2f} "
        f"max={arr.max():.2f} | worst |Δobj|={worst_obj:.4f} | worst gt_err={worst_gt:.1f}px"
    )
    # Int8 weight quantization shifts corner points by ~2 in-px on average
    # (256x256 input grid); the fp32 model's own error on these synthetic
    # pages (gt_err) is an order of magnitude larger, so this is the noise floor.
    assert arr.mean() < 3.0, "mean backend corner disagreement"
    assert arr.max() < 8.0, "worst backend corner disagreement"
    assert worst_obj < 0.25, "has_obj disagreement"


if __name__ == "__main__":
    main()
