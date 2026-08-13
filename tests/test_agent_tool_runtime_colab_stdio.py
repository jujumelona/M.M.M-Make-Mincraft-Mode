from __future__ import annotations

import io

import anyio
import pytest

from minecraft_mod_ai import agent_tool_runtime


class _NotebookStderr(io.StringIO):
    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


class _FakeStdioContext:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self):
        self.events.append("stdio-enter")
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
        self.events.append("stdio-exit")
        return False


class _FakeClientSession:
    def __init__(self, _read, _write, *, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self):
        self.events.append("session-enter")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.events.append("session-exit")
        return False

    async def initialize(self) -> None:
        self.events.append("initialize")


def test_mcp_stdio_does_not_pass_notebook_stderr_to_subprocess(monkeypatch) -> None:
    import mcp
    import mcp.client.stdio as mcp_stdio

    events: list[str] = []
    notebook_stderr = _NotebookStderr()
    monkeypatch.setattr(agent_tool_runtime.sys, "stderr", notebook_stderr)

    def fake_stdio_client(_params, *, errlog):
        assert errlog is not notebook_stderr
        assert isinstance(errlog.fileno(), int)
        events.append("fd-backed-errlog")
        return _FakeStdioContext(events)

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(
        mcp,
        "ClientSession",
        lambda read, write: _FakeClientSession(read, write, events=events),
    )

    async def exercise() -> None:
        async with agent_tool_runtime._MCPStdioSession(
            stage="planning",
            env={},
            timeout_seconds=5.0,
        ) as session:
            assert isinstance(session, _FakeClientSession)

    anyio.run(exercise)

    assert events == [
        "fd-backed-errlog",
        "stdio-enter",
        "session-enter",
        "initialize",
        "session-exit",
        "stdio-exit",
    ]


def test_failed_stdio_enter_preserves_original_exception(monkeypatch) -> None:
    import mcp.client.stdio as mcp_stdio

    events: list[str] = []

    class SpawnFailure(RuntimeError):
        pass

    class FailingStdioContext:
        async def __aenter__(self):
            events.append("stdio-enter")
            raise SpawnFailure("original spawn failure")

        async def __aexit__(self, exc_type, exc, tb):
            events.append("stdio-exit")
            raise AssertionError("__aexit__ must not run when __aenter__ failed")

    def fake_stdio_client(_params, *, errlog):
        assert isinstance(errlog.fileno(), int)
        return FailingStdioContext()

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client)

    session = agent_tool_runtime._MCPStdioSession(
        stage="planning",
        env={},
        timeout_seconds=5.0,
    )
    with pytest.raises(SpawnFailure, match="original spawn failure"):
        anyio.run(session.__aenter__)

    assert events == ["stdio-enter"]


def test_initialize_failure_is_not_masked_by_cleanup_failure(monkeypatch) -> None:
    import mcp
    import mcp.client.stdio as mcp_stdio

    events: list[str] = []

    class InitFailure(ValueError):
        pass

    class CleanupFailingSession(_FakeClientSession):
        async def initialize(self) -> None:
            self.events.append("initialize")
            raise InitFailure("initialize failed")

        async def __aexit__(self, exc_type, exc, tb):
            self.events.append("session-exit")
            raise RuntimeError("cleanup also failed")

    def fake_stdio_client(_params, *, errlog):
        assert isinstance(errlog.fileno(), int)
        return _FakeStdioContext(events)

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(
        mcp,
        "ClientSession",
        lambda read, write: CleanupFailingSession(read, write, events=events),
    )

    session = agent_tool_runtime._MCPStdioSession(
        stage="planning",
        env={},
        timeout_seconds=5.0,
    )
    with pytest.raises(InitFailure, match="initialize failed") as caught:
        anyio.run(session.__aenter__)

    assert "cleanup also failed" in "\n".join(getattr(caught.value, "__notes__", ()))
    assert events == [
        "stdio-enter",
        "session-enter",
        "initialize",
        "session-exit",
        "stdio-exit",
    ]
