from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager

import anyio

from minecraft_mod_ai import mcp_transport_pool
from minecraft_mod_ai.mcp_child_trace_contract import traced_stdio_session


def test_runtime_uses_visible_child_stderr_session_factory() -> None:
    kwdefaults = mcp_transport_pool.MCPTransportPool.__init__.__kwdefaults__ or {}

    assert kwdefaults["session_factory"] is traced_stdio_session
    assert getattr(mcp_transport_pool, "_mmm_child_trace_contract_installed", False) is True


def test_nonblocking_transport_hot_path_is_not_rebound_by_trace_contract() -> None:
    execute = mcp_transport_pool.MCPTransportPool._execute

    # Child visibility is installed through the session-factory default only. Keep the
    # reviewed non-blocking _execute mutation as the sole transport method rebind.
    assert getattr(execute, "_mmm_nonblocking_transport_execute_v1", False) is True
    assert execute.__module__ == "minecraft_mod_ai.runtime_hot_path_contract"


def test_stdio_child_stderr_is_forwarded_to_parent_stderr(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class FakeParams:
        def __init__(self, *, command, args, env):
            captured["command"] = command
            captured["args"] = args
            captured["env"] = env

    class FakeSession:
        def __init__(self, read_stream, write_stream):
            captured["streams"] = (read_stream, write_stream)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return {"server": "fake"}

    @asynccontextmanager
    async def fake_stdio_client(params, *, errlog):
        captured["errlog"] = errlog
        errlog.write("child-stderr-visible\n")
        errlog.flush()
        yield "read", "write"

    fake_mcp = types.ModuleType("mcp")
    fake_mcp.ClientSession = FakeSession
    fake_mcp.StdioServerParameters = FakeParams
    fake_client = types.ModuleType("mcp.client")
    fake_stdio = types.ModuleType("mcp.client.stdio")
    fake_stdio.stdio_client = fake_stdio_client
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", fake_client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", fake_stdio)

    async def run() -> None:
        async with traced_stdio_session(
            "generation",
            {"MMM_MCP_STAGE": "generation"},
            2.0,
        ):
            pass

    anyio.run(run)
    stderr = capsys.readouterr().err

    assert captured["errlog"] is not None
    assert "child-stderr-visible" in stderr
    assert '"event":"mcp_transport_session_start"' in stderr
    assert '"event":"mcp_transport_initialized"' in stderr
    assert '"stderr_route":"parent_stderr"' in stderr
