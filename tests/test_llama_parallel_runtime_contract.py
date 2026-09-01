from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import minecraft_mod_ai.model_router as router_module
import minecraft_mod_ai.scheduler_parallel_safety_contract as scheduler_module
from minecraft_mod_ai.llama_parallel_runtime_contract import ReentrantReadWriteLock
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
        return SimpleNamespace(content='{"status":"tool-aware"}', tool_calls=())

    def generate(self, request):
        self.generate_requests.append(request)
        return '{"game_design":{}}'


def test_parallel_runtime_contract_is_installed() -> None:
    assert getattr(ModelRouter.generate_text, "_mmm_llama_shared_slots", False)
    assert getattr(ModelRouter.generate_text, "_mmm_preserves_agent_tools", False)
    assert getattr(ModelRouter.generate_text, "_mmm_preserves_response_schema", False)
    assert getattr(ModelRouter.generate_text, "_mmm_uses_canonical_request_preparation", False)
    assert getattr(ModelRouter.generate_text, "_mmm_parallel_router_contract_version", 0) >= 3
    assert getattr(ModelRouter.generation_session, "_mmm_llama_shared_slots", False)
    assert getattr(scheduler_module._capacities, "_mmm_dynamic_llama_slots", False)
    assert hasattr(router_module, "_LLAMA_INFERENCE_SLOTS")


def test_scheduler_llm_capacity_follows_selected_native_slots(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    assert scheduler_module._capacities()["llm"] == 4
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert scheduler_module._capacities()["llm"] == 1


def test_parallel_router_keeps_one_stable_selector_owned_tool_surface(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    monkeypatch.setenv("MMM_AGENT_TOOLS", "1")
    router = ModelRouter(profile="t4_local").bind_agent_workspace(tmp_path)
    adapter = _ToolAwareAdapter()
    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)

    tool_result = router.generate_text(
        "coder",
        ({"role": "user", "content": "repair"},),
        response_format="json",
        tool_stage="generation",
        enable_tools=True,
    )
    assert tool_result == '{"status":"tool-aware"}'
    assert len(adapter.turn_requests) == 1
    request = adapter.turn_requests[0]
    exposed_names = {str(tool["function"]["name"]) for tool in request.tools}

    assert exposed_names
    assert {"search_code_rag", "search_project_rag"} <= exposed_names
    assert "plan_complete_game" not in exposed_names
    assert "runtime_start_server" not in exposed_names
    assert "package_release" not in exposed_names
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is True
    assert request.response_format == "json"

    capability_messages = [
        str(message.get("content", ""))
        for message in request.messages
        if message.get("role") == "system"
        and "mmm/agent-capability-context-v5" in str(message.get("content", ""))
    ]
    assert capability_messages
    capability = capability_messages[-1]
    assert "ground-production-with-live-evidence" in capability
    assert "mmm/agent-capability-context-v5" in capability

    schema = {
        "type": "object",
        "properties": {"game_design": {"type": "object"}},
        "required": ["game_design"],
    }
    plain_result = router.generate_text(
        "coder",
        ({"role": "user", "content": "repair without tools"},),
        response_format="json",
        response_schema=schema,
        tool_stage="generation",
        enable_tools=False,
    )
    assert plain_result == '{"game_design":{}}'
    assert len(adapter.generate_requests) == 1
    assert adapter.generate_requests[0].response_schema == schema


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
    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)
    results: list[str] = []

    def run() -> None:
        results.append(router.generate_text("planner", ({"role": "user", "content": "x"},), enable_tools=False))

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
    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)
    results: list[str] = []

    def run() -> None:
        results.append(router.generate_text("planner", ({"role": "user", "content": "x"},), enable_tools=False))

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
