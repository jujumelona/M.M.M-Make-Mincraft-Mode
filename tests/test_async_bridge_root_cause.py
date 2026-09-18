from __future__ import annotations

import pytest

from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime
from minecraft_mod_ai.external_mcp_router import ExternalMCPRouter


async def _raise_real_failure() -> None:
    raise ValueError("real provider failure")


def test_agent_runtime_loop_probe_does_not_mask_real_failure() -> None:
    runtime = object.__new__(AgentToolRuntime)

    with pytest.raises(ValueError, match="real provider failure") as captured:
        runtime._run_async(_raise_real_failure)

    context = captured.value.__context__
    assert context is None or "no running event loop" not in str(context)


def test_external_router_loop_probe_does_not_mask_real_failure(
    monkeypatch,
) -> None:
    router = ExternalMCPRouter(timeout_seconds=1.0)

    async def fail_provider(server_name, entry, *, tool, arguments):
        raise ValueError("real external provider failure")

    monkeypatch.setattr(router, "_call_provider_async", fail_provider)

    with pytest.raises(ValueError, match="real external provider failure") as captured:
        router._call_provider("provider", {}, tool="lookup", arguments={})

    context = captured.value.__context__
    assert context is None or "no running event loop" not in str(context)
