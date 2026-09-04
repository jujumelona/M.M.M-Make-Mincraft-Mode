from __future__ import annotations

import io
import sys
import tempfile
import types
from contextlib import asynccontextmanager
from typing import Any

import anyio

from minecraft_mod_ai import mcp_child_trace_contract, mcp_transport_pool
from minecraft_mod_ai.mcp_child_trace_contract import traced_stdio_session


class _NotebookStderr:
    """IPython/Colab-like text stream: writable but not subprocess-compatible."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def write(self, value: str) -> int:
        return self.buffer.write(value)

    def flush(self) -> None:
        return None

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


def _install_fake_mcp(monkeypatch, *, write_child_marker: bool = False) -> dict[str, Any]:
    captured: dict[str, Any] = {}

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
        captured["errlog_fileno"] = errlog.fileno()
        if write_child_marker:
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
    return captured


def _run_traced_session() -> None:
    async def run() -> None:
        async with traced_stdio_session(
            "generation",
            {"MMM_MCP_STAGE": "generation"},
            2.0,
        ):
            pass

    anyio.run(run)


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


def test_probe_fileno_records_notebook_unsupported_operation() -> None:
    descriptor, failure = mcp_child_trace_contract._probe_fileno(_NotebookStderr())

    assert descriptor is None
    assert failure == "UnsupportedOperation: fileno"


def test_subprocess_stderr_uses_original_stderr_when_notebook_stream_has_no_fd(
    monkeypatch,
) -> None:
    notebook_stderr = _NotebookStderr()
    with tempfile.TemporaryFile(mode="w+b") as original_stderr:
        monkeypatch.setattr(mcp_child_trace_contract.sys, "stderr", notebook_stderr)
        monkeypatch.setattr(mcp_child_trace_contract.sys, "__stderr__", original_stderr)

        with mcp_child_trace_contract._subprocess_stderr_target() as (
            target,
            route,
            failures,
        ):
            assert target is original_stderr
            assert route == "parent_dunder_stderr"
            assert target.fileno() >= 0
            assert failures == {"parent_stderr": "UnsupportedOperation: fileno"}


def test_subprocess_stderr_duplicates_fd2_when_python_streams_have_no_fd(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mcp_child_trace_contract.sys, "stderr", _NotebookStderr())
    monkeypatch.setattr(mcp_child_trace_contract.sys, "__stderr__", _NotebookStderr())

    with mcp_child_trace_contract._subprocess_stderr_target() as (
        target,
        route,
        failures,
    ):
        assert route == "parent_fd2_duplicate"
        assert target.fileno() >= 0
        assert failures == {
            "parent_stderr": "UnsupportedOperation: fileno",
            "parent_dunder_stderr": "UnsupportedOperation: fileno",
        }


def test_traced_session_never_passes_notebook_stderr_to_mcp_subprocess(monkeypatch) -> None:
    captured = _install_fake_mcp(monkeypatch)
    notebook_stderr = _NotebookStderr()

    with tempfile.TemporaryFile(mode="w+b") as original_stderr:
        monkeypatch.setattr(mcp_child_trace_contract.sys, "stderr", notebook_stderr)
        monkeypatch.setattr(mcp_child_trace_contract.sys, "__stderr__", original_stderr)

        _run_traced_session()

        assert captured["errlog"] is original_stderr
        assert captured["errlog_fileno"] == original_stderr.fileno()
        trace = notebook_stderr.buffer.getvalue()
        assert '"event":"mcp_transport_session_start"' in trace
        assert '"stderr_route":"parent_dunder_stderr"' in trace
        assert '"parent_stderr":"UnsupportedOperation: fileno"' in trace


def test_stdio_child_stderr_is_forwarded_to_parent_stderr(monkeypatch, capfd) -> None:
    captured = _install_fake_mcp(monkeypatch, write_child_marker=True)

    _run_traced_session()
    stderr = capfd.readouterr().err

    assert captured["errlog"] is not None
    assert captured["errlog_fileno"] >= 0
    assert "child-stderr-visible" in stderr
    assert '"event":"mcp_transport_session_start"' in stderr
    assert '"event":"mcp_transport_initialized"' in stderr
    assert '"stderr_route":"parent_stderr"' in stderr
