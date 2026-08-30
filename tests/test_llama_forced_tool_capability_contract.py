from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from minecraft_mod_ai import forced_tool_execution_contract as forced


def _reset() -> None:
    with forced._NATIVE_PROBE_LOCK:
        forced._NATIVE_PROBE_CACHE.clear()
        forced._NATIVE_PROBE_NEGATIVE_AT.clear()
        forced._NATIVE_PROBE_TRANSIENT_AT.clear()
        forced._NATIVE_PROBE_KEY_LOCKS.clear()


def _adapter():
    return SimpleNamespace(
        config=SimpleNamespace(model_id="model"),
        _server_url=lambda request: "http://127.0.0.1:8080",
    )


def _success():
    return SimpleNamespace(
        tool_calls=(SimpleNamespace(name=forced._NATIVE_PROBE_TOOL, arguments={"nonce": "mmm"}),)
    )


def test_transient_failure_cools_down_without_poisoning(monkeypatch) -> None:
    _reset()
    monkeypatch.setattr(forced, "_native_probe_request", lambda request: request)
    monkeypatch.setenv("MMM_LLAMA_NATIVE_TOOL_TRANSIENT_COOLDOWN_SECONDS", "5")
    times = iter((0.0, 2.0, 6.0))
    monkeypatch.setattr(forced.time, "monotonic", lambda: next(times))
    calls = 0

    def current(_adapter, _request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary transport failure")
        return _success()

    adapter, request = _adapter(), object()
    assert forced._native_required_supported(current, adapter, request) is False
    assert forced._NATIVE_PROBE_CACHE == {}
    assert forced._native_required_supported(current, adapter, request) is False
    assert calls == 1
    assert forced._native_required_supported(current, adapter, request) is True
    assert calls == 2


def test_protocol_negative_expires(monkeypatch) -> None:
    _reset()
    monkeypatch.setattr(forced, "_native_probe_request", lambda request: request)
    monkeypatch.setenv("MMM_LLAMA_NATIVE_TOOL_NEGATIVE_TTL_SECONDS", "60")
    times = iter((0.0, 30.0, 61.0))
    monkeypatch.setattr(forced.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(forced, "_native_protocol_failure", lambda exc: True)
    calls = 0

    def current(_adapter, _request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("protocol rejected")
        return _success()

    adapter, request = _adapter(), object()
    assert forced._native_required_supported(current, adapter, request) is False
    assert forced._native_required_supported(current, adapter, request) is False
    assert calls == 1
    assert forced._native_required_supported(current, adapter, request) is True
    assert calls == 2


def test_mark_native_unsupported_is_ttl_scoped(monkeypatch) -> None:
    _reset()
    monkeypatch.setattr(forced.time, "monotonic", lambda: 10.0)
    adapter, request = _adapter(), object()
    forced._mark_native_unsupported(adapter, request)
    key = ("http://127.0.0.1:8080", "model")
    assert forced._NATIVE_PROBE_CACHE[key] is False
    assert forced._NATIVE_PROBE_NEGATIVE_AT[key] == 10.0


def test_concurrent_probe_is_deduplicated_per_endpoint(monkeypatch) -> None:
    _reset()
    monkeypatch.setattr(forced, "_native_probe_request", lambda request: request)
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(4)

    def current(_adapter, _request):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.03)
        return _success()

    adapter, request = _adapter(), object()
    results: list[bool] = []

    def run() -> None:
        barrier.wait()
        results.append(forced._native_required_supported(current, adapter, request))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True] * 4
    assert calls == 1
