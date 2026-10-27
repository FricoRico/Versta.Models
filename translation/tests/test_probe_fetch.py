"""
Tests for the pinned-bundle fetch hardening (NEU-60, failure classification
NEU-148): transient CDN errors (5xx/429/timeouts) are retried with bounded
exponential backoff, non-retryable answers (404) fail fast, and exhausted
retries surface as a distinct, labeled cdn-fetch-flake failure (exit 2 +
::error annotation) so pack-acceptance reviews can tell infrastructure from
pack signal.
"""

import json
import sys

from pathlib import Path
from typing import Callable, Dict, List
from urllib.error import HTTPError, URLError

import pytest

from versta.probe import fetch
from versta.probe.fetch import (
    EXIT_CDN_FETCH_FLAKE,
    KIND_ARTIFACT_MISSING,
    KIND_CDN_FETCH_FLAKE,
    fetch_published_bundles,
    main,
)

from conftest import FakeResponse

BASE = "https://models.versta.app/translation/v2.0.0"
PAYLOAD = b"published-pack-bytes"


def _http_error(code: int) -> HTTPError:
    return HTTPError(f"{BASE}/xx-en-bundle.tar.gz", code, f"HTTP {code}", {}, None)


def _write_pins(tmp_path: Path, slugs: List[str]) -> Path:
    pins = {
        "published": {
            "bundle_base_url": BASE,
            "directions": {},
            "bundles": {
                f"{slug}-bundle.tar.gz": {
                    "sha256": "0" * 64,
                    "url": f"{BASE}/{slug}-bundle.tar.gz",
                }
                for slug in slugs
            },
        }
    }
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")
    return pins_path


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[str], FakeResponse]
) -> List[str]:
    calls: List[str] = []

    def fake(url: str, timeout: int = 0) -> FakeResponse:
        calls.append(url)
        return handler(url)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return calls


def test_forced_503_is_retried_with_backoff_and_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance (NEU-148): a forced 503 on one bundle fetch is retried and
    the fetch completes."""
    attempts: Dict[str, int] = {}

    def flaky(url: str) -> FakeResponse:
        attempts[url] = attempts.get(url, 0) + 1
        if url.endswith("-bundle.tar.gz") and attempts[url] <= 2:
            raise _http_error(503)
        return FakeResponse(PAYLOAD)

    calls = _fake_urlopen(monkeypatch, flaky)
    sleeps: List[float] = []
    monkeypatch.setattr("versta.probe.fetch.time.sleep", sleeps.append)

    pins_path = _write_pins(tmp_path, ["xx-en"])
    missing = fetch_published_bundles(pins_path, tmp_path / "out")

    assert missing == []
    assert (tmp_path / "out" / "xx-en-bundle.tar.gz").read_bytes() == PAYLOAD
    tar_calls = [c for c in calls if c.endswith("-bundle.tar.gz")]
    assert len(tar_calls) == 3
    # Exponential backoff between the three attempts (attempt 3 needs no wait).
    assert sleeps == [fetch.FETCH_BACKOFF_SECONDS, fetch.FETCH_BACKOFF_SECONDS * 2]


def test_timeout_is_retried_and_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def flaky(url: str) -> FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise URLError(TimeoutError("timed out"))
        return FakeResponse(PAYLOAD)

    _fake_urlopen(monkeypatch, flaky)
    monkeypatch.setattr(fetch, "FETCH_BACKOFF_SECONDS", 0.0)

    pins_path = _write_pins(tmp_path, ["xx-en"])
    missing = fetch_published_bundles(pins_path, tmp_path / "out")

    assert missing == []
    assert (tmp_path / "out" / "xx-en-bundle.tar.gz").read_bytes() == PAYLOAD


def test_exhausted_retries_are_labeled_cdn_fetch_flake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance (NEU-148): exhausted retries produce a distinct, labeled
    failure — kind cdn-fetch-flake, exit 2, ::error annotation."""
    monkeypatch.setattr(fetch, "FETCH_BACKOFF_SECONDS", 0.0)

    def down(url: str) -> FakeResponse:
        raise _http_error(503)

    _fake_urlopen(monkeypatch, down)
    pins_path = _write_pins(tmp_path, ["xx-en"])

    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch", "--pins", str(pins_path), "--output_dir", str(tmp_path / "out")],
    )
    exit_code = main()

    assert exit_code == EXIT_CDN_FETCH_FLAKE
    assert exit_code != 1

    report = json.loads((tmp_path / "out" / "fetch-report.json").read_text())
    assert len(report["missing"]) == 2  # tar + .sha256 sidecar
    assert {entry["kind"] for entry in report["missing"]} == {KIND_CDN_FETCH_FLAKE}
    assert {entry["attempts"] for entry in report["missing"]} == {fetch.FETCH_TRIES}


def test_exhausted_retries_emit_error_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch, "FETCH_BACKOFF_SECONDS", 0.0)

    def down(url: str) -> FakeResponse:
        raise _http_error(503)

    _fake_urlopen(monkeypatch, down)
    pins_path = _write_pins(tmp_path, ["xx-en"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch", "--pins", str(pins_path), "--output_dir", str(tmp_path / "out")],
    )
    assert main() == EXIT_CDN_FETCH_FLAKE

    out = capsys.readouterr().out
    assert f"::error title={KIND_CDN_FETCH_FLAKE}::" in out
    assert "transient CDN infrastructure failure, not a pack/product regression" in out


def test_404_fails_fast_as_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pack that is not served (404) is a deliberate defect signal: no
    retries, kind artifact-missing, generic exit 1."""
    monkeypatch.setattr(fetch, "FETCH_BACKOFF_SECONDS", 0.0)

    def gone(url: str) -> FakeResponse:
        raise _http_error(404)

    calls = _fake_urlopen(monkeypatch, gone)
    pins_path = _write_pins(tmp_path, ["xx-en"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch", "--pins", str(pins_path), "--output_dir", str(tmp_path / "out")],
    )
    exit_code = main()

    assert exit_code == 1
    assert len(calls) == 2  # one attempt each for tar + sidecar
    report = json.loads((tmp_path / "out" / "fetch-report.json").read_text())
    assert {entry["kind"] for entry in report["missing"]} == {KIND_ARTIFACT_MISSING}
    assert {entry["attempts"] for entry in report["missing"]} == {1}


def test_mixed_flake_and_missing_artifact_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "FETCH_BACKOFF_SECONDS", 0.0)

    def mixed(url: str) -> FakeResponse:
        if "yy-en" in url:
            raise _http_error(404)
        raise _http_error(503)

    _fake_urlopen(monkeypatch, mixed)
    pins_path = _write_pins(tmp_path, ["xx-en", "yy-en"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch", "--pins", str(pins_path), "--output_dir", str(tmp_path / "out")],
    )

    assert main() == 1
    report = json.loads((tmp_path / "out" / "fetch-report.json").read_text())
    kinds = {entry["bundle"]: entry["kind"] for entry in report["missing"]}
    assert kinds["xx-en-bundle.tar.gz"] == KIND_CDN_FETCH_FLAKE
    assert kinds["yy-en-bundle.tar.gz"] == KIND_ARTIFACT_MISSING


def test_tolerate_failures_reports_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch, "FETCH_BACKOFF_SECONDS", 0.0)

    def down(url: str) -> FakeResponse:
        raise _http_error(503)

    _fake_urlopen(monkeypatch, down)
    pins_path = _write_pins(tmp_path, ["xx-en"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch",
            "--pins",
            str(pins_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--tolerate_failures",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert f"::warning title={KIND_CDN_FETCH_FLAKE}::" in out


def test_cached_files_are_kept_and_report_still_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "xx-en-bundle.tar.gz").write_bytes(PAYLOAD)
    (out_dir / "xx-en-bundle.tar.sha256").write_bytes(b"sha256-sidecar")

    def unreachable(url: str) -> FakeResponse:
        raise AssertionError(f"unexpected fetch of {url}")

    _fake_urlopen(monkeypatch, unreachable)
    pins_path = _write_pins(tmp_path, ["xx-en"])

    assert fetch_published_bundles(pins_path, out_dir) == []
    report = json.loads((out_dir / "fetch-report.json").read_text())
    assert report == {"missing": []}
