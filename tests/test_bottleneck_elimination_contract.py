from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai import agentic_pre_design_rag as pre_design_rag
from minecraft_mod_ai import bottleneck_elimination_contract as bottlenecks
from minecraft_mod_ai import external_mcp_router
from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime


def test_json_tracker_stops_only_after_complete_root_object() -> None:
    tracker = bottlenecks._JsonObjectTracker()

    assert not tracker.feed('  {"message":"brace } and escaped quote \\"')
    assert not tracker.feed(' stays in string","nested":{"value":1}')
    assert tracker.feed(',"items":[1,2,3]}   ')
    assert tracker.complete is True
    assert tracker.invalid is False

    rendered = (
        '  {"message":"brace } and escaped quote \\"'
        ' stays in string","nested":{"value":1},"items":[1,2,3]}   '
    )
    assert isinstance(json.loads(rendered), dict)


def test_json_tracker_rejects_trailing_or_non_object_output() -> None:
    trailing = bottlenecks._JsonObjectTracker()
    assert not trailing.feed('{"ok":true} trailing')
    assert trailing.invalid is True

    array_root = bottlenecks._JsonObjectTracker()
    assert not array_root.feed('[1,2,3]')
    assert array_root.invalid is True


def test_auto_agentic_search_does_not_serialize_best_of_n_on_one_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    risky = {
        "current_target_deliverables": ["a", "b", "c"],
        "request": "networking multiplayer custom_java migration persistence",
    }

    assert agentic._planner_candidate_count(risky, "production planning") == 1

    # Explicit search remains an opt-in quality override.
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    assert agentic._planner_candidate_count(risky, "production planning") == 3


def test_small_rag_units_are_packed_without_dropping_unit_identity() -> None:
    evidence = {f"source_{index}": {"value": index} for index in range(12)}
    packed = list(pre_design_rag._evidence_units(evidence))

    assert len(packed) < len(evidence)
    recovered: set[str] = set()
    for unit_id, value in packed:
        if unit_id.startswith("packed:"):
            assert isinstance(value, dict)
            for record in value["units"]:
                recovered.add(str(record["unit_id"]))
        else:
            recovered.add(unit_id)
    assert recovered == set(evidence)


def test_external_read_calls_single_flight_identical_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bottlenecks._READ_CACHE.clear()
    bottlenecks._READ_INFLIGHT.clear()

    class Worker:
        server_info = {"name": "fake"}

        def __init__(self) -> None:
            self.calls = 0
            self.lock = threading.Lock()

        def call_tool(self, tool: str, arguments: dict[str, object]) -> dict[str, object]:
            del tool, arguments
            with self.lock:
                self.calls += 1
            time.sleep(0.05)
            return {"value": 7}

    worker = Worker()
    monkeypatch.setattr(bottlenecks, "_external_worker", lambda *args, **kwargs: worker)
    router = external_mcp_router.ExternalMCPRouter(timeout_seconds=2.0)
    entry = {
        "transport": "stdio",
        "capabilities": {"docs": {"tool": "search", "access": "read"}},
    }

    def call() -> dict[str, object]:
        return router._call_provider(
            "fake-server",
            entry,
            tool="search",
            arguments={"query": "same"},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _index: call(), range(4)))

    assert worker.calls == 1
    assert [item["result"] for item in results] == [{"value": 7}] * 4


def test_external_providers_are_not_globally_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bottlenecks._READ_CACHE.clear()
    bottlenecks._READ_INFLIGHT.clear()
    barrier = threading.Barrier(2)

    class Worker:
        server_info = {"name": "fake"}

        def call_tool(self, tool: str, arguments: dict[str, object]) -> dict[str, object]:
            del tool, arguments
            barrier.wait(timeout=1.0)
            return {"ok": True}

    workers = {"server-a": Worker(), "server-b": Worker()}
    monkeypatch.setattr(
        bottlenecks,
        "_external_worker",
        lambda _router, server_name, _entry: workers[server_name],
    )
    router = external_mcp_router.ExternalMCPRouter(timeout_seconds=2.0)
    entry = {
        "transport": "stdio",
        "capabilities": {"docs": {"tool": "search", "access": "read"}},
    }

    def call(server_name: str) -> dict[str, object]:
        return router._call_provider(
            server_name,
            entry,
            tool="search",
            arguments={"query": server_name},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(call, "server-a")
        future_b = pool.submit(call, "server-b")
        assert future_a.result()["result"] == {"ok": True}
        assert future_b.result()["result"] == {"ok": True}


def test_first_party_tool_runtime_routes_list_and_call_through_persistent_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Worker:
        def __init__(self) -> None:
            self.list_count = 0
            self.call_count = 0

        def list_tools(self):
            self.list_count += 1
            return ({"name": "x", "description": "", "input_schema": {}},)

        def call_tool(self, name: str, arguments: dict[str, object]):
            self.call_count += 1
            return {"name": name, "arguments": arguments}

    worker = Worker()
    monkeypatch.setattr(bottlenecks, "_first_party_worker", lambda *args, **kwargs: worker)
    runtime = AgentToolRuntime(profile="fast_test", timeout_seconds=2.0)

    listed = runtime._run_async(runtime._list_tools_async, "research")
    called = runtime._run_async(runtime._call_tool_async, "research", "x", {"a": 1})

    assert listed[0]["name"] == "x"
    assert called == {"name": "x", "arguments": {"a": 1}}
    assert worker.list_count == 1
    assert worker.call_count == 1
