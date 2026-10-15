"""
Package tunables for the Supertonic 3 pipeline, with calibration rationale.

Grouped here per the shared conventions (no scattered per-module globals);
steady-state values change with the data that justifies them, not ad hoc.
"""

from typing import Dict, List

from .typing import MODEL_STAGES

#: Numerics gate for the shipped int8 set. Calibrated to the NEU-64 spike
#: table (2026-10-14): int8 vs fp32 diverges LSD ≈ 7.5–15 dB on wav across
#: 8 languages (worst: ar 14.9, spectral r 0.820; cosine 0.836 typical).
#: Thresholds sit just above the worst observation with safety headroom;
#: anything materially worse than the spike envelope fails here, not in QA.
INT8_GATE: Dict[str, float] = {
    "min_snr_db": 5.0,
    "min_cosine": 0.80,
    "max_lsd_db": 16.0,
    "min_spectral_r": 0.80,
}

#: Converter-correctness control: fp32 MNN vs the fp32 ONNX reference must
#: be near-exact, or the converter — not the quantization — is broken.
FP32_CONTROL_GATE: Dict[str, float] = {"min_snr_db": 60.0, "min_cosine": 0.9999}

#: fp16 is recorded informationally only (chaotic diffusion trajectories
#: legitimately reach 12–15 dB on fr/ko per the spike); it never gates.

#: Published tarball budget (2/6 scope): 120 MB for the int8 bundle.
BUNDLE_SIZE_CAP_BYTES: int = 120 * 1024 * 1024

#: The bundle's complete file inventory (app model root). Lint fails closed
#: on any absence — no partial bundle may be produced.
REQUIRED_BUNDLE_FILES: List[str] = (
    [f"{stage}.mnn" for stage in MODEL_STAGES]
    + ["tts.json", "unicode_indexer.json", "LICENSE", "README.md", "metadata.json"]
    + [
        f"voice_styles/{name}.json"
        for name in ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
    ]
)

#: Wav analysis framing for the LSD/spectral metrics (44.1 kHz output).
LSD_FRAME_SIZE: int = 2048
LSD_HOP_SIZE: int = 512
