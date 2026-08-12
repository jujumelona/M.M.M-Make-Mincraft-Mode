from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import minecraft_mod_ai.complete_planner as complete_planner_module
import minecraft_mod_ai.model_router as router_module
import minecraft_mod_ai.scheduler_parallel_safety_contract as scheduler_module
from minecraft_mod_ai.llama_parallel_runtime_contract import (
    ReentrantReadWriteLock,
    _planner_parallel_capacity,
)
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def load_profile(self, name):
        return object()

    def role(self, profile, role):
        return SimpleNamespace(
            role=role,
            provider="local",
            adapter="llama_cpp",
            exclusive_gpu=True,
        )


class _BlockingAdapter:
    def __init__(self, barrier: threading.Barrier, events: list[tuple[str, float]]) -> None:
        self.barrier = barrier
        self.events = events

    def generate(self, request):
        self.events.append(("start", time.monotonic()))
        self.barrier.wait(timeout=2)
        time.sleep(0.05)
        self.events.append(("end", time.monotonic()))
        return "ok"


class _SlotProbeAdapter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.two_entered = threading.Event()
        self.started = 0
        self.active = 0
        self.max_active = 0

    def generate(self, request):
        with self.lock:
            self.started += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= 2:
                self.two_entered.set()
        self.release.wait(timeout=2)
        with self.lock:
            self.active -= 1
        return "ok"


class _ToolAwareAdapter:
    def __init__(self) -> None:
        self.turn_requests = []
        self.generate_requests = []

    def generate_turn(self, request):
        self.turn_requests.append(request)
        return SimpleNamespace(content="tool-aware", tool_calls=())

    def generate(self, request):
        self.generate_requests.append(request)
        return "plain"


class _ToolRuntime:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def tool_schemas(self, stage):
        self.stages.append(stage)
        return (
            {
                "type": "function",
                "function": {
                    "name": "inspect_project",
                    "description": "inspect",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )


class _PlannerProbeRouter:
    profile = "test"

    def __init__(self) -> None:
        self.registry = _Registry()
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def generate_text(
        self,
        role,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ):
        assert role == "planner"
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=2)
            system = str(messages[0]["content"])
            candidate_two = "Candidate 2 of 2" in system
            payload = {
                "modules": [
                    {
                        "module_id": "candidate_two" if candidate_two else "candidate_one",
                        "config": {
                            "requirement_refs": ["request:test"] if candidate_two else [],
                        },
                        "depends_on": ["core"] if candidate_two else [],
                    }
                ],
                "acceptance_tests": ["verified"] if candidate_two else [],
                "completed_deliverables": ["feature"] if candidate_two else [],
            }
            return json.dumps(payload)
        finally:
            with self.lock:
                self.active -= 1


def test_parallel_runtime_contract_is_installed() -> None:
    assert getattr(ModelRouter.generate_text, "_mmm_llama_shared_slots", False)
    assert getattr(ModelRouter.generate_text, "_mmm_preserves_agent_tools", False)
    assert getattr(ModelRouter.generation_session, "_mmm_llama_shared_slots", False)
    assert getattr(scheduler_module._capacities, "_mmm_dynamic_llama_slots", False)
    assert getattr(
        complete_planner_module._generate_json_page_with_repair,
        "_mmm_parallel_plan_search",
        False,
    )
    assert hasattr(router_module, "_LLAMA_INFERENCE_SLOTS")


def test_scheduler_llm_capacity_follows_selected_native_slots(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    assert scheduler_module._capacities()["llm"] == 4
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert scheduler_module._capacities()["llm"] == 1


def test_planner_parallel_capacity_is_native_local_only(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    router = SimpleNamespace(profile="test", registry=_Registry())
    assert _planner_parallel_capacity(router, 2) == 2

    class _RemoteRegistry(_Registry):
        def role(self, profile, role):
            value = super().role(profile, role)
            value.provider = "remote"
            return value

    remote = SimpleNamespace(profile="test", registry=_RemoteRegistry())
    assert _planner_parallel_capacity(remote, 2) == 1


def test_parallel_router_preserves_stage_tools_and_enable_tools(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    runtime = _ToolRuntime()
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_kwargs: runtime,
    )
    adapter = _ToolAwareAdapter()
    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)

    tool_result = router.generate_text(
        "coder",
        ({"role": "user", "content": "repair"},),
        response_format="json",
        tool_stage="generation",
        enable_tools=True,
    )

    assert tool_result == "tool-aware"
    assert runtime.stages == ["generation"]
    assert len(adapter.turn_requests) == 1
    request = adapter.turn_requests[0]
    assert request.tools
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is True
    assert request.response_format == "json"

    plain_result = router.generate_text(
        "coder",
        ({"role": "user", "content": "repair without tools"},),
        tool_stage="generation",
        enable_tools=False,
    )
    assert plain_result == "plain"
    assert len(adapter.generate_requests) == 1
    assert runtime.stages == ["generation"]


def test_verified_planner_candidates_use_native_slots(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _PlannerProbeRouter()

    result = complete_planner_module._generate_json_page_with_repair(
        router,
        system_prompt="plan",
        request={"scope": "multiplayer networking persistence migration"},
        media_paths=(),
        expected_contracts=(
            frozenset({"modules", "acceptance_tests", "completed_deliverables"}),
        ),
        stage="production page",
    )

    assert router.calls == 2
    assert router.max_active == 2
    assert result["modules"][0]["module_id"] == "candidate_two"


def test_shared_gpu_lock_allows_readers_but_blocks_writer() -> None:
    lock = ReentrantReadWriteLock()
    entered = threading.Barrier(3)
    release = threading.Event()
    writer_entered = threading.Event()

    def reader() -> None:
        with lock.shared():
            entered.wait(timeout=2)
            release.wait(timeout=2)

    readers = [threading.Thread(target=reader) for _ in range(2)]
    for thread in readers:
        thread.start()
    entered.wait(timeout=2)

    def writer() -> None:
        with lock:
            writer_entered.set()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    time.sleep(0.05)
    assert not writer_entered.is_set()
    release.set()
    for thread in readers:
        thread.join(timeout=2)
    writer_thread.join(timeout=2)
    assert writer_entered.is_set()


def test_same_router_llama_requests_overlap_when_parallel_selected(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = ModelRouter(profile="test", registry=_Registry())
    barrier = threading.Barrier(2)
    events: list[tuple[str, float]] = []
    adapter = _BlockingAdapter(barrier, events)
    monkeypatch.setattr(
        router,
        "_new_text_adapter",
        lambda config, role: adapter,
    )

    results: list[str] = []

    def run() -> None:
        results.append(
            router.generate_text(
                "planner",
                ({"role": "user", "content": "x"},),
            )
        )

    threads = [threading.Thread(target=run) for _ in range(2)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    elapsed = time.monotonic() - started

    assert results == ["ok", "ok"]
    assert len([event for event, _ in events if event == "start"]) == 2
    assert elapsed < 0.5


def test_direct_router_calls_never_exceed_native_llama_slots(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = ModelRouter(profile="test", registry=_Registry())
    adapter = _SlotProbeAdapter()
    monkeypatch.setattr(
        router,
        "_new_text_adapter",
        lambda config, role: adapter,
    )

    results: list[str] = []

    def run() -> None:
        results.append(
            router.generate_text(
                "planner",
                ({"role": "user", "content": "x"},),
            )
        )

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()

    assert adapter.two_entered.wait(timeout=2)
    time.sleep(0.05)
    with adapter.lock:
        assert adapter.started == 2
        assert adapter.max_active == 2

    adapter.release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert results == ["ok", "ok", "ok"]
    assert adapter.started == 3
    assert adapter.max_active == 2
