"""
The hermetic-suite guard (NEU-53): proves the PR-path test run cannot touch
the network, so CI stays fast, deterministic and independent of upstream
state. Real-network model evaluation lives in the release-candidate
workflow only.
"""

import socket

from urllib.request import urlopen

import pytest

from conftest import NetworkBlockedError

# Documentation range (RFC 5737): unroutable even if the guard regresses, so
# these tests fail fast instead of hanging on a real connection attempt.
_BLOCKED_ADDRESS = ("192.0.2.1", 443)


def test_create_connection_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(_BLOCKED_ADDRESS, timeout=1)


def test_socket_connect_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(_BLOCKED_ADDRESS)


def test_urlopen_boundary_is_blocked() -> None:
    # The versta.download boundary: a test that forgets to mock urlopen must
    # hit the guard, not the network.
    with pytest.raises(NetworkBlockedError):
        urlopen("https://example.invalid/registry.json", timeout=1)
