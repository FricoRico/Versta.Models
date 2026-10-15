"""
CLI dispatcher for the Supertonic 3 pipeline.

    uv run python -m versta.supertonic3 convert --upstream-revision <sha>
    uv run python -m versta.supertonic3 verify --snapshot <dir> --converted <dir> --upstream-revision <sha>
    uv run python -m versta.supertonic3 bundle --converted <dir> --version v2.0.0
    uv run python -m versta.supertonic3 record --converted <dir> --bundle <tarball> --version v2.0.0
    uv run python -m versta.supertonic3 feed --version v2.0.0 --bundle <tarball>
"""

import sys

from . import build_bundle, convert_to_mnn, feed, record_ledger, verify_numerics

COMMANDS = {
    "convert": convert_to_mnn.main,
    "verify": verify_numerics.main,
    "bundle": build_bundle.main,
    "record": record_ledger.main,
    "feed": feed.main,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(
            f"usage: python -m versta.supertonic3 <{'|'.join(COMMANDS)}> [args]",
            file=sys.stderr,
        )
        sys.exit(2)
    command = sys.argv[1]
    sys.argv = [f"versta.supertonic3 {command}", *sys.argv[2:]]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
