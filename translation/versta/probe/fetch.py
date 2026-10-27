"""
Fetch the pinned published bundles (NEU-60 hardening of the pack-acceptance
workflows, failure classification added in NEU-148): every served
``<slug>-bundle.tar.gz`` + ``.sha256`` sidecar, with bounded retry/backoff,
because a single transient 503 over 64 fetches must not kill a 90-minute
arm64 sweep.

Failures are classified so a red check tells reviewers which kind of signal
they are looking at:

- ``cdn-fetch-flake`` — every attempt hit a *transient* error (HTTP
  408/429/5xx, timeout, DNS/connection reset). Infrastructure flake; safe to
  re-run. In strict mode the CLI annotates these as ``::error
  title=cdn-fetch-flake::`` and exits ``EXIT_CDN_FETCH_FLAKE`` (2) when they
  are the only failures.
- ``artifact-missing`` — the CDN answered non-retryably (e.g. 404). The pack
  is not being served: a deliberate defect/release signal, never retried.

Used by both the pack-acceptance workflow (strict: any missing fetch fails
the job) and the weekly sweep (``--tolerate_failures``: missing bundles are
reported as defects instead — a pack that stopped being served is a defect).
"""

import json
import time

from pathlib import Path
from typing import List, TypedDict
from urllib.error import HTTPError, URLError

from .pins import load_published_pins

FETCH_TRIES = 3
FETCH_BACKOFF_SECONDS = 5.0

# Worth a retry: rate limiting and server-side/transient statuses. The
# observed flake class is CDN 503s that clear within seconds; anything
# client-side (404, 403, …) is a deliberate "not served" signal instead.
RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 509})

KIND_CDN_FETCH_FLAKE = "cdn-fetch-flake"
KIND_ARTIFACT_MISSING = "artifact-missing"

# Strict-mode exit code when every missing fetch is a transient flake. The
# pack-acceptance workflow turns it into a labeled step summary; any
# artifact-missing entry keeps the generic exit 1.
EXIT_CDN_FETCH_FLAKE = 2


class MissEntry(TypedDict):
    """One pinned file that could not be fetched, with its failure class."""

    bundle: str
    url: str
    kind: str
    attempts: int
    error: str


class FetchError(RuntimeError):
    """
    A single pinned file could not be downloaded.

    `kind` is KIND_CDN_FETCH_FLAKE when the bounded retries were exhausted on
    transient errors, KIND_ARTIFACT_MISSING when the server answered
    non-retryably (no retry attempted — fail fast on a real defect signal).
    """

    def __init__(self, message: str, kind: str, attempts: int) -> None:
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in RETRYABLE_HTTP_STATUSES
    # DNS failures, connection resets and read/connect timeouts arrive as
    # URLError (reason holds the socket error); socket.timeout is TimeoutError
    # on py3.10+. All are transient at the CDN boundary.
    return isinstance(error, (URLError, TimeoutError, ConnectionError))


def _fetch_one(url: str, destination: Path) -> None:
    from urllib.request import urlopen

    delay = FETCH_BACKOFF_SECONDS
    last_error: Exception = URLError("no attempt made")
    for attempt in range(1, FETCH_TRIES + 1):
        try:
            with urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:  # classified below; boundary is network-only
            if not _is_retryable(error):
                raise FetchError(
                    f"{url}: {error}", KIND_ARTIFACT_MISSING, attempt
                ) from error
            last_error = error
            if attempt == FETCH_TRIES:
                break
            print(
                f"transient fetch error (attempt {attempt}/{FETCH_TRIES}), "
                f"retrying in {delay:.0f}s: {error}"
            )
            time.sleep(delay)
            delay *= 2
    raise FetchError(
        f"{url}: exhausted {FETCH_TRIES} attempts on transient errors: {last_error}",
        KIND_CDN_FETCH_FLAKE,
        FETCH_TRIES,
    ) from last_error


def fetch_published_bundles(pins_path: Path, output_dir: Path) -> List[MissEntry]:
    """
    Downloads every bundle pinned under the ``published`` section of
    pack-acceptance/pins.json. Existing files are kept (CI caches them).

    Args:
        pins_path (Path): pack-acceptance/pins.json.
        output_dir (Path): where ``*-bundle.tar.gz`` + sidecars land.

    Returns:
        List[MissEntry]: entries that could not be fetched — empty when
            everything was served. Also written to
            ``<output_dir>/fetch-report.json``.
    """
    published = load_published_pins(pins_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: List[MissEntry] = []
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
                missing.append(
                    MissEntry(
                        bundle=name,
                        url=url,
                        kind=error.kind,
                        attempts=error.attempts,
                        error=str(error),
                    )
                )
                print(f"MISSING {filename} [{error.kind}]: {error}")
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
    if not missing:
        return 0

    level = "warning" if args.tolerate_failures else "error"
    for entry in missing:
        if entry["kind"] != KIND_CDN_FETCH_FLAKE:
            continue
        print(
            f"::{level} title={KIND_CDN_FETCH_FLAKE}::{entry['url']} did not "
            f"serve after {entry['attempts']} attempts — transient CDN "
            "infrastructure failure, not a pack/product regression. "
            "Safe to re-run the job."
        )

    if args.tolerate_failures:
        print(
            f"{len(missing)} pinned bundle file(s) could not be fetched "
            "(tolerated; the weekly sweep files them as defects)"
        )
        return 0

    permanent = [m for m in missing if m["kind"] != KIND_CDN_FETCH_FLAKE]
    if permanent:
        print(
            f"{len(permanent)} pinned bundle file(s) are not served "
            f"({KIND_ARTIFACT_MISSING}): deliberate defect/release signal"
        )
        return 1

    print(
        f"{KIND_CDN_FETCH_FLAKE}: all {len(missing)} missing fetch(es) "
        "exhausted retries on transient CDN errors — infrastructure failure, "
        "distinct from pack signal"
    )
    return EXIT_CDN_FETCH_FLAKE


if __name__ == "__main__":
    raise SystemExit(main())
