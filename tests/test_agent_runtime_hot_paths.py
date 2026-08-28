from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import anyio

import minecraft_mod_ai.agent_capability_context as capability_context
import minecraft_mod_ai.runtime_hotpath_consolidation as hotpath
from minecraft_mod_ai import agent_tool_runtime


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_first_party_execution_does_not_repeat_schema_listing(monkeypatch) -> None:
    runtime = agent_tool_runtime.AgentToolRuntime(profile="test")
    events: list[str] = []

    class Session:
        async def list_tools(self):
            raise AssertionError("execution hot path repeated the stage schema listing")

        async def call_tool(self, name, *, arguments):
            events.append(f"execute:{name}:{arguments['query']}")
            return SimpleNamespace(isError=False, structuredContent={"ok": True}, content=())

    class Context:
        async def __aenter__(self):
            events.append("enter")
            return Session()

        async def __aexit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    monkeypatch.setattr(runtime, "_session", lambda _stage: Context())
    result = anyio.run(
        runtime._call_tool_async,
        "generation",
        "search_code_rag",
        {"query": "block registry"},
    )
    assert result["structured_content"] == {"ok": True}
    assert events == ["enter", "execute:search_code_rag:block registry", "exit"]


def test_manifest_router_is_reused_but_request_target_stays_dynamic(monkeypatch) -> None:
    instances: list[object] = []
    targets: list[dict[str, str]] = []

    class FakeRouter:
        def __init__(self) -> None:
            instances.append(self)

        def capability_manifest(self, *, stage, target, max_access):
            targets.append(dict(target))
            return {"capabilities": {}}

    capability_context._manifest_router.cache_clear()
    monkeypatch.setattr(capability_context, "ExternalMCPRouter", FakeRouter)
    schemas = (_schema("external_mcp_capabilities"),)
    monkeypatch.setenv("MMM_MINECRAFT_VERSION", "1.21.1")
    capability_context.build_agent_capability_context("research", schemas, model_role="researcher")
    monkeypatch.setenv("MMM_MINECRAFT_VERSION", "1.21.4")
    capability_context.build_agent_capability_context("research", schemas, model_role="researcher")
    assert len(instances) == 1
    assert [target["minecraft_version"] for target in targets] == ["1.21.1", "1.21.4"]
    capability_context._manifest_router.cache_clear()


def test_central_research_identical_inflight_reads_are_single_flight() -> None:
    provider_calls = 0
    provider_lock = threading.Lock()
    first_provider_entered = threading.Event()
    duplicate_provider_entered = threading.Event()
    release_provider = threading.Event()
    launch = threading.Barrier(2)

    def provider(query: str, **kwargs):
        nonlocal provider_calls
        with provider_lock:
            provider_calls += 1
            if provider_calls > 1:
                duplicate_provider_entered.set()
        first_provider_entered.set()
        if not release_provider.wait(timeout=2.0):
            raise TimeoutError("test provider was not released")
        return {"query": query, "kwargs": dict(kwargs)}

    def retrieve_domain_evidence(_brief, *, retrieve=provider):
        def fetch():
            launch.wait(timeout=2.0)
            return retrieve("same-query", source="official")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fetch) for _ in range(2)]
            return {"rows": [future.result(timeout=2.0) for future in futures]}

    module = SimpleNamespace(
        retrieve_domain_evidence=retrieve_domain_evidence,
        retrieve_official_evidence=provider,
    )
    wrapped = hotpath._install_central_research_dedup(module)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result_future = executor.submit(wrapped, {"need": "evidence"})
        assert first_provider_entered.wait(timeout=1.0)
        try:
            assert not duplicate_provider_entered.wait(timeout=0.2)
            assert provider_calls == 1
        finally:
            release_provider.set()
        result = result_future.result(timeout=2.0)
    assert result["rows"][0] == result["rows"][1]
    assert provider_calls == 1


def test_project_scoped_lock_reclaims_inactive_project_entries() -> None:
    lock = hotpath._ProjectScopedRLock()
    for index in range(64):
        token = hotpath._MEMORY_BASE.set(f"/tmp/mmm-project-{index}")
        try:
            with lock:
                assert len(lock._locks) == 1
                with lock:
                    assert len(lock._locks) == 1
                assert len(lock._locks) == 1
        finally:
            hotpath._MEMORY_BASE.reset(token)
        assert lock._locks == {}


def test_runtime_close_releases_only_materialized_transport_pool() -> None:
    runtime = agent_tool_runtime.AgentToolRuntime(profile="test")
    events: list[str] = []

    class Pool:
        def close(self) -> None:
            events.append("pool-close")

    class Finalizer:
        alive = True

        def detach(self) -> None:
            events.append("finalizer-detach")
            self.alive = False

    pool = Pool()
    finalizer = Finalizer()
    runtime._mcp_transport_pool = pool
    runtime._mcp_transport_pool_finalizer = finalizer
    runtime.close()

    assert events == ["finalizer-detach", "pool-close"]
    assert runtime._mcp_transport_pool is None
    assert runtime._mcp_transport_pool_finalizer is None
    runtime.close()
    assert events == ["finalizer-detach", "pool-close"]
