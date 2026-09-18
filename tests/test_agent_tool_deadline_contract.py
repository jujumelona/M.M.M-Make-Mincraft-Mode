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
