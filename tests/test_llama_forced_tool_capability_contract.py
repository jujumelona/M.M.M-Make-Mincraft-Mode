from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai import llama_forced_tool_capability_contract as contract


def _fake_module(*, protocol_failure: bool):
    cache: dict[tuple[str, str], bool] = {}
    lock = threading.RLock()

    def contains_exact_call(turn, name):
        calls = tuple(getattr(turn, "tool_calls", ()) or ())
        return len(calls) == 1 and getattr(calls[0], "name", "") == name

    module = SimpleNamespace(
        _NATIVE_PROBE_CACHE=cache,
        _NATIVE_PROBE_LOCK=lock,
        _NATIVE_PROBE_TOOL="mmm_required_tool_probe",
        _native_required_supported=lambda current, adapter, request: False,
        _mark_native_unsupported=lambda adapter, request: cache.__setitem__(("url", "model"), False),
        _native_probe_key=lambda adapter, request: ("url", "model"),
        _native_probe_cache_key=lambda adapter, request: ("url", "model"),
        _native_probe_request=lambda request: request,
        _contains_exact_call=contains_exact_call,
        _native_protocol_failure=lambda exc: protocol_failure,
    )
    contract.install(module)
    return module


def _successful_turn():
    return SimpleNamespace(
        tool_calls=(
            SimpleNamespace(
                name="mmm_required_tool_probe",
                arguments={"nonce": "mmm"},
            ),
        )
    )


def test_transient_probe_failure_does_not_poison_native_capability(monkeypatch) -> None:
    module = _fake_module(protocol_failure=False)
    times = iter((0.0, 2.0, 6.0))
    monkeypatch.setattr(contract.time, "monotonic", lambda: next(times))
    calls = 0

    def current(_adapter, _request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary transport failure")
        return _successful_turn()

    adapter = object()
    request = object()
    assert module._native_required_supported(current, adapter, request) is False
    assert module._NATIVE_PROBE_CACHE == {}
    assert module._native_required_supported(current, adapter, request) is False
    assert calls == 1
    assert module._native_required_supported(current, adapter, request) is True
    assert calls == 2
    assert module._NATIVE_PROBE_CACHE[("url", "model")] is True


def test_protocol_negative_capability_expires_and_is_reprobed(monkeypatch) -> None:
    module = _fake_module(protocol_failure=True)
    times = iter((0.0, 30.0, 61.0))
    monkeypatch.setattr(contract.time, "monotonic", lambda: next(times))
    calls = 0

    def current(_adapter, _request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("native required-tool protocol rejected")
        return _successful_turn()

    adapter = object()
    request = object()
    assert module._native_required_supported(current, adapter, request) is False
    assert module._NATIVE_PROBE_CACHE[("url", "model")] is False
    assert module._native_required_supported(current, adapter, request) is False
    assert calls == 1
    assert module._native_required_supported(current, adapter, request) is True
    assert calls == 2
    assert module._NATIVE_PROBE_CACHE[("url", "model")] is True


def test_explicit_native_protocol_failure_mark_is_ttl_scoped(monkeypatch) -> None:
    module = _fake_module(protocol_failure=True)
    monkeypatch.setattr(contract.time, "monotonic", lambda: 10.0)
    module._mark_native_unsupported(object(), object())

    key = ("url", "model")
    assert module._NATIVE_PROBE_CACHE[key] is False
    assert module._mmm_native_probe_negative_at[key] == 10.0


def test_runtime_installs_recoverable_forced_tool_capability_policy() -> None:
    assert getattr(
        forced._native_required_supported,
        "_mmm_recoverable_native_tool_probe_v1",
        False,
    )
    assert getattr(
        forced._mark_native_unsupported,
        "_mmm_ttl_native_tool_negative_v1",
        False,
    )
