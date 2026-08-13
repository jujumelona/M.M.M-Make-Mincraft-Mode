from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

from minecraft_mod_ai import model_router
from minecraft_mod_ai import small_model_agent_policy as small_policy
from minecraft_mod_ai import small_model_research_contract as frontier


def test_semantic_single_flight_ignores_object_identity() -> None:
    target = SimpleNamespace(
        _planner_key=lambda prompt, brief: (prompt, id(brief)),
        _ecosystem_key=lambda prompt, design, brief: (prompt, id(design), id(brief)),
    )
    frontier._install_semantic_single_flight(target)

    brief_a = {"domains": [{"id": "fabric", "queries": ["api", "mapping"]}]}
    brief_b = {"domains": [{"queries": ["api", "mapping"], "id": "fabric"}]}
    design_a = {"modules": [{"id": "x", "depends_on": []}]}
    design_b = {"modules": [{"depends_on": [], "id": "x"}]}

    assert target._planner_key("same", brief_a) == target._planner_key("same", brief_b)
    assert target._ecosystem_key("same", design_a, brief_a) == target._ecosystem_key(
        "same", design_b, brief_b
    )

    brief_b["domains"][0]["queries"].append("different")
    assert target._planner_key("same", brief_a) != target._planner_key("same", brief_b)


@dataclass(frozen=True)
class _Call:
    name: str
    id: str


def test_mixed_tool_batch_runs_parallel_read_waves_around_serial_barrier(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_PARALLEL_READS", "4")
    calls = (
        _Call("search_code_rag", "a"),
        _Call("search_project_rag", "b"),
        _Call("external_mcp_call", "barrier"),
        _Call("inspect_existing_mod", "c"),
        _Call("quality_status", "d"),
    )

    first_wave = threading.Barrier(2)
    lock = threading.Lock()
    active_reads = 0
    peak_reads = 0
    barrier_active_reads: list[int] = []

    def execute(call: _Call):
        nonlocal active_reads, peak_reads
        is_read = call.name in model_router._PARALLEL_READ_TOOLS
        if is_read:
            with lock:
                active_reads += 1
                peak_reads = max(peak_reads, active_reads)
            if call.id in {"a", "b"}:
                first_wave.wait(timeout=2.0)
            with lock:
                active_reads -= 1
        else:
            with lock:
                barrier_active_reads.append(active_reads)
        return call, {"ok": True, "tool": call.name}

    executed = model_router._execute_tool_waves(calls, execute)

    assert [call.id for call, _payload in executed] == [call.id for call in calls]
    assert peak_reads >= 2
    assert barrier_active_reads == [0]


def test_verified_workflow_history_controls_search_breadth(monkeypatch) -> None:
    monkeypatch.setattr(
        small_policy,
        "_matches",
        lambda *_args, **_kwargs: [
            {"similarity": 0.93, "recovered_from": []},
            {"similarity": 0.88, "recovered_from": []},
            {"similarity": 0.84, "recovered_from": []},
        ],
    )
    assert small_policy.planner_search_width_hint(
        "networking integration", "planning", maximum=3
    ) == 1

    monkeypatch.setattr(
        small_policy,
        "_matches",
        lambda *_args, **_kwargs: [
            {"similarity": 0.91, "recovered_from": ["schema", "dependency"]},
            {"similarity": 0.79, "recovered_from": ["evidence"]},
        ],
    )
    assert small_policy.planner_search_width_hint(
        "networking integration", "planning", maximum=3
    ) == 3


def test_trace_adaptive_width_escalates_and_safely_suppresses(monkeypatch) -> None:
    target = SimpleNamespace(
        _planner_candidate_count=lambda _request, _stage: 2,
        _mode=lambda: "auto",
        _env_int=lambda *_args, **_kwargs: 3,
    )
    frontier._install_trace_adaptive_search(target)

    monkeypatch.setattr(
        small_policy,
        "planner_search_width_hint",
        lambda *_args, **_kwargs: 3,
    )
    assert target._planner_candidate_count({"description": "simple"}, "planning") == 3

    monkeypatch.setattr(
        small_policy,
        "planner_search_width_hint",
        lambda *_args, **_kwargs: 1,
    )
    assert target._planner_candidate_count(
        {"description": "networking integration"}, "planning"
    ) == 1

    maximal = {
        "description": "networking integration " + ("x" * (13 * 1024)),
        "current_target_deliverables": ["a", "b", "c"],
    }
    assert target._planner_candidate_count(maximal, "planning") == 2
