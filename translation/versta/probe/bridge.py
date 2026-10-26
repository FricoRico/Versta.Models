"""
The bridge between an installed bundle direction and the leanmt engine:
invokes the pinned ``leanmt-probe`` binary with the exact field mapping the
app's JNI layer applies (Versta.Android ``app/native/jni/src/leanmt.cc`` and
``bridge/leanmt/LeanmtModel.kt``): model/vocabulary/target_vocabulary/
shortlist paths plus encoder_layers/decoder_layers/ffn_depth/num_heads from
the direction metadata, with app-default decode (beam 4, max sequence
length 256) and the translation cache disabled for measurement.

Deliberately a subprocess boundary rather than bindings: the acceptance
runner must measure the process-level peak RSS of one model load + probe
translating, and a crashed load (exit 3) is a scoreable outcome, not a
runner failure. JNI-surface parity is coordinated with the app pair — if the
JNI argument list changes, this module changes in the same PR.
"""

import json
import subprocess

from pathlib import Path
from typing import List, Optional, cast

from .config import ProbeConfig
from .install import InstalledDirection
from .typing import ProbeOutput


class BridgeError(RuntimeError):
    """The probe binary could not be executed or produced unreadable output."""


def probe_command(
    probe_bin: Path,
    direction: InstalledDirection,
    probes_file: Path,
    json_out: Path,
    config: ProbeConfig = ProbeConfig(),
) -> List[str]:
    """
    Builds the leanmt-probe invocation for one installed direction, mirroring
    the JNI argument order for readability in CI logs.

    Args:
        probe_bin (Path): path of the built leanmt-probe binary.
        direction (InstalledDirection): the installed direction to probe.
        probes_file (Path): source sentences, one per line.
        json_out (Path): where leanmt-probe writes its JSON payload.
        config (ProbeConfig): decode settings (app defaults when unset).

    Returns:
        List[str]: the full argv.
    """
    argv = [
        str(probe_bin),
        "--model",
        str(direction.model),
        "--vocab",
        str(direction.vocabulary),
    ]
    # Shared-vocabulary models omit this flag; the JNI layer passes the empty
    # string in the same situation.
    if direction.target_vocabulary != Path(""):
        argv += ["--target_vocab", str(direction.target_vocabulary)]
    argv += [
        "--shortlist",
        str(direction.shortlist),
        "--encoder_layers",
        str(direction.encoder_layers),
        "--decoder_layers",
        str(direction.decoder_layers),
        "--ffn_depth",
        str(direction.ffn_depth),
        "--num_heads",
        str(direction.num_heads),
        "--probes",
        str(probes_file),
        "--beam",
        str(config.beam),
        "--max_seq",
        str(config.max_sequence_length),
        "--json",
        str(json_out),
    ]
    if config.threads > 0:
        argv += ["--threads", str(config.threads)]
    return argv


def run_probe(
    probe_bin: Path,
    direction: InstalledDirection,
    sentences: List[str],
    work_dir: Path,
    config: ProbeConfig = ProbeConfig(),
) -> ProbeOutput:
    """
    Runs leanmt-probe for one direction over the given corpus sentences.

    Args:
        probe_bin (Path): path of the built leanmt-probe binary.
        direction (InstalledDirection): the installed direction to probe.
        sentences (List[str]): corpus source sentences, verbatim.
        work_dir (Path): scratch directory for the probe input/output files.
        config (ProbeConfig): decode settings (app defaults when unset).

    Returns:
        ProbeOutput: parsed leanmt-probe JSON (load_ok False included — a
            failed load is a measurement, not an exception here).

    Raises:
        BridgeError: the binary is missing/not executable, or it exited in a
            way that still left no parseable JSON.
    """
    if not probe_bin.is_file():
        raise BridgeError(f"leanmt-probe binary not found: {probe_bin}")

    safe = f"{direction.source_language}-{direction.target_language}"
    probes_file = work_dir / f"{safe}.probes.txt"
    json_out = work_dir / f"{safe}.probe.json"
    probes_file.write_text("\n".join(sentences) + "\n", encoding="utf-8")

    completed = subprocess.run(
        probe_command(probe_bin, direction, probes_file, json_out, config),
        capture_output=True,
        text=True,
    )
    # Exit codes: 0 ok, 2 bad args (harness bug — raise), 3 load failure
    # (scoreable), 4 a probe failed to translate (scoreable).
    if completed.returncode == 2:
        raise BridgeError(
            f"leanmt-probe rejected our invocation (exit 2):\n{completed.stderr}"
        )

    payload: Optional[str] = None
    if json_out.is_file():
        payload = json_out.read_text(encoding="utf-8")
    elif completed.stdout.strip().startswith("{"):
        payload = completed.stdout
    if payload is None:
        raise BridgeError(
            f"leanmt-probe exited {completed.returncode} with no JSON payload; "
            f"stderr: {completed.stderr[-2000:]}"
        )

    try:
        return cast(ProbeOutput, json.loads(payload))
    except json.JSONDecodeError as error:
        raise BridgeError(f"leanmt-probe produced unparseable JSON: {error}") from error
