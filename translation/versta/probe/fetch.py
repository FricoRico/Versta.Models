"""
Fetch the pinned published bundles (NEU-60 hardening of the pack-acceptance
workflows): every served ``<slug>-bundle.tar.gz`` + ``.sha256`` sidecar,
with retry/backoff, because a single transient 503 over 64 fetches must not
kill a 90-minute arm64 sweep.

Used by both the pack-acceptance workflow (strict: any missing fetch fails
the job) and the weekly sweep (``--tolerate_failures``: missing bundles are
reported as defects instead — a pack that stopped being served is a defect).
"""

import json
import time

from pathlib import Path
from typing import Dict, List
from urllib.error import URLError

from .pins import load_published_pins

FETCH_TRIES = 3
FETCH_BACKOFF_SECONDS = 2.0


class FetchError(RuntimeError):
    """Strict fetch: at least one pinned bundle could not be downloaded."""


def _fetch_one(url: str, destination: Path) -> None:
    from urllib.request import urlopen

    delay = FETCH_BACKOFF_SECONDS
    last_error: Exception = URLError("no attempt made")
    for attempt in range(1, FETCH_TRIES + 1):
        try:
            with urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:  # HTTP, DNS, socket — all retryable here
            last_error = error
            if attempt == FETCH_TRIES:
                break
            time.sleep(delay)
            delay *= 2
    raise FetchError(f"{url}: {last_error}")


def fetch_published_bundles(pins_path: Path, output_dir: Path) -> List[Dict[str, str]]:
    """
    Downloads every bundle pinned under the ``published`` section of
    pack-acceptance/pins.json. Existing files are kept (CI caches them).

    Args:
        pins_path (Path): pack-acceptance/pins.json.
        output_dir (Path): where ``*-bundle.tar.gz`` + sidecars land.

    Returns:
        List[Dict[str, str]]: entries that could not be fetched, as
            ``{"bundle": name, "url": url, "error": message}`` — empty when
            everything was served.
    """
    published = load_published_pins(pins_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: List[Dict[str, str]] = []
    for name, pin in sorted(published["bundles"].items()):
        for url in (pin["url"], pin["url"].replace(".tar.gz", ".tar.sha256")):
            filename = url.rsplit("/", 1)[-1]
            target = output_dir / filename
            if target.exists():
                continue
            try:
                _fetch_one(url, target)
                print(f"fetched {filename}")
            except FetchError as error:
                missing.append({"bundle": name, "url": url, "error": str(error)})
                print(f"MISSING {filename}: {error}")
    report_path = output_dir / "fetch-report.json"
    report_path.write_text(
        json.dumps({"missing": missing}, indent=2) + "\n", encoding="utf-8"
    )
    return missing


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch the pinned published bundles with retry/backoff."
    )
    parser.add_argument("--pins", type=Path, default=Path("pack-acceptance/pins.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/published"))
    parser.add_argument(
        "--tolerate_failures",
        action="store_true",
        help="Report missing bundles (exit 0) instead of failing the run; "
        "the weekly sweep files missing fetches as defects.",
    )
    args = parser.parse_args()

    missing = fetch_published_bundles(args.pins, args.output_dir)
    if missing and not args.tolerate_failures:
        print(f"{len(missing)} pinned bundle file(s) could not be fetched")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
