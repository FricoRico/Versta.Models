"""
Append the shipped bundle to the checked-in conversion ledger.

This is the deliberate human seam between "a tarball exists" and "a release
exists": it refuses to record anything that failed the numerics gate unless
an explicit override reason is supplied, and it raises (never overwrites) on a
bundle sha256 already recorded. The ledger entry is self-contained — source
revision, toolchain, per-artifact hashes, the full verification report and
the shipped bundle hash ride together, so the ledger alone answers "what
shipped, from what, proven how?".
"""

import argparse
import json

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .convert_to_mnn import sha256_file
from .ledger import append_entry, ledger_path
from .typing import LedgerEntry, UpstreamRecord, VerificationReport

MODULE_ROOT = Path(__file__).resolve().parents[2]


def record(
    converted_dir: Path,
    bundle_tarball: Path,
    version: str,
    ledger_file: Path,
    override_reason: Optional[str] = None,
) -> Path:
    """
    Appends one shipped-version record to the ledger.

    Args:
        converted_dir (Path): Conversion output dir (needs `conversion-record.json`).
        bundle_tarball (Path): The tarball produced by `bundle`.
        version (str): Bundle version, e.g. "v2.0.0".
        ledger_file (Path): The checked-in ledger document.
        override_reason (str|None): Required if, and only if, the
            verification report did not pass — makes overrides loud.

    Raises:
        FileNotFoundError: If the conversion record or verification report is missing.
        LedgerError: If the gate failed without an override reason, or the
            bundle is already recorded.
    """
    with open(converted_dir / "conversion-record.json", encoding="utf-8") as handle:
        conversion = json.load(handle)
    with open(converted_dir / "verification.json", encoding="utf-8") as handle:
        verification: VerificationReport = json.load(handle)
    if verification["passed"] and override_reason:
        raise ValueError("override_reason only applies to a failed gate")
    if not verification["passed"] and not override_reason:
        raise ValueError(
            "verification gate failed; refusing to record without an explicit "
            "override_reason describing why this ships anyway"
        )
    upstream = UpstreamRecord(
        repo=conversion["upstream"]["repo"], revision=conversion["upstream"]["revision"]
    )
    entry = LedgerEntry(
        recorded_at=datetime.now(timezone.utc).isoformat(),
        version=version,
        upstream=upstream,
        toolchain=conversion["toolchain"],
        artifacts=conversion["artifacts"],
        verification=verification,
        bundle_sha256=sha256_file(bundle_tarball),
        bundle_bytes=bundle_tarball.stat().st_size,
        override_reason=override_reason,
    )
    return append_entry(ledger_file, entry)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        prog="versta.supertonic3 record",
        description="Append the shipped bundle to the checked-in conversion ledger.",
    )
    parser.add_argument(
        "--converted",
        type=Path,
        required=True,
        help="Conversion output dir (from `convert`, with verify's JSON report).",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="The tarball produced by `bundle`.",
    )
    parser.add_argument("--version", required=True, help="Bundle version, e.g. v2.0.0.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ledger_path(MODULE_ROOT),
        help="Ledger document; defaults to the checked-in evals/ledger file.",
    )
    parser.add_argument(
        "--override_reason",
        default=None,
        help="Non-empty reason, required when the verification gate failed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record(args.converted, args.bundle, args.version, args.ledger, args.override_reason)


if __name__ == "__main__":
    main()
