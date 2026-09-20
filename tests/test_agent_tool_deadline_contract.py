from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import model_router
from minecraft_mod_ai.deadline_executor import ParallelExecutionTimeout


def _blocking_execute(call):
    time.sleep(0.50)
    return call, {"ok": True, "tool": call.name}


@pytest.mark.parametrize("name", ["search_code_rag", "apply_source_edit"])
def test_every_agent_tool_wave_has_an_outer_deadline(monkeypatch, name: str) -> None:
    monkeypatch.setattr(model_router, "_agent_tool_timeout_seconds", lambda: 0.05)
    call = SimpleNamespace(name=name, arguments={})

    started = time.monotonic()
    with pytest.raises(ParallelExecutionTimeout):
        model_router._execute_tool_waves((call,), _blocking_execute)

    assert time.monotonic() - started < 0.30



def test_java_verifier_outer_deadline_cannot_undercut_cold_start_budget(
    monkeypatch,
) -> None:
    from minecraft_mod_ai import generation_verifier_resilience

    monkeypatch.setattr(model_router, "_agent_tool_timeout_seconds", lambda: 120.0)
    monkeypatch.setattr(model_router, "_agent_tool_return_grace_seconds", lambda: 30.0)
    monkeypatch.setattr(
        generation_verifier_resilience,
        "host_jdt_startup_timeout_seconds",
        lambda: 300,
    )
    call = SimpleNamespace(
        name="java_diagnostics",
        arguments={"timeout_seconds": 90},
    )

    assert model_router._parallel_read_call(call) is False
    assert model_router._agent_tool_call_timeout_seconds(call) == 330.0


def test_java_verifier_outer_deadline_still_bounds_hung_worker(monkeypatch) -> None:
    from minecraft_mod_ai import generation_verifier_resilience

    monkeypatch.setattr(model_router, "_agent_tool_timeout_seconds", lambda: 0.02)
    monkeypatch.setattr(model_router, "_agent_tool_return_grace_seconds", lambda: 0.02)
    monkeypatch.setattr(
        generation_verifier_resilience,
        "host_jdt_startup_timeout_seconds",
        lambda: 0.05,
    )
    call = SimpleNamespace(
        name="java_diagnostics",
        arguments={"timeout_seconds": 0.01},
    )

    started = time.monotonic()
    with pytest.raises(ParallelExecutionTimeout):
        model_router._execute_tool_waves((call,), _blocking_execute)

    elapsed = time.monotonic() - started
    assert 0.04 <= elapsed < 0.30
