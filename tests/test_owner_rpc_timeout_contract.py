from __future__ import annotations

import threading
from typing import Any

from minecraft_mod_ai.owner_rpc import OwnerRPC


class _FakeStdin:
    def write(self, _value: str) -> None:
        return None

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()


class _RecordingResponses:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def get(self, *, timeout: float) -> dict[str, Any]:
        self.timeouts.append(timeout)
        return {'id': 1, 'result': {}}


def _make_rpc() -> tuple[OwnerRPC, _RecordingResponses]:
    rpc = object.__new__(OwnerRPC)
    responses = _RecordingResponses()
    rpc.process = _FakeProcess()
    rpc._responses = responses
    rpc._lock = threading.RLock()
    rpc._closed = False
    rpc._sequence = 0
    return rpc, responses


def test_owner_rpc_preserves_legacy_90_second_caller_budget() -> None:
    rpc, responses = _make_rpc()

    assert rpc.request('resolve', {}, timeout=90.0) == {}
    assert responses.timeouts == [90.0]


def test_owner_rpc_honors_explicit_600_second_hard_budget() -> None:
    rpc, responses = _make_rpc()

    assert rpc.request('resolve', {}, timeout=600.0) == {}
    assert responses.timeouts == [600.0]


def test_owner_rpc_caps_timeout_above_transport_safety_ceiling() -> None:
    rpc, responses = _make_rpc()

    assert rpc.request('resolve', {}, timeout=900.0) == {}
    assert responses.timeouts == [600.0]
