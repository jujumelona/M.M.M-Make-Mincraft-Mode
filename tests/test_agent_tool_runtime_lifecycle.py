from __future__ import annotations

from threading import RLock

from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime


class _Resource:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    def close(self) -> None:
        self.closed.append(self.name)


def _runtime() -> AgentToolRuntime:
    runtime = object.__new__(AgentToolRuntime)
    runtime._lock = RLock()
    return runtime


def test_close_owns_generation_jdt_lifecycle_directly() -> None:
    closed: list[str] = []
    runtime = _runtime()
    runtime._mmm_generation_java_service = _Resource("jdt", closed)

    runtime.close()

    assert closed == ["jdt"]
    assert not hasattr(runtime, "_mmm_generation_java_service")


def test_close_releases_all_materialized_persistent_resources() -> None:
    closed: list[str] = []
    runtime = _runtime()
    runtime._mmm_generation_java_service = _Resource("jdt", closed)
    runtime._mcp_transport_pool = _Resource("mcp", closed)
    runtime._mcp_transport_pool_finalizer = None

    runtime.close()
    runtime.close()

    assert closed == ["jdt", "mcp"]
    assert not hasattr(runtime, "_mmm_generation_java_service")
    assert runtime._mcp_transport_pool is None
