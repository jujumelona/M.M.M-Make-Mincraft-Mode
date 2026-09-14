from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.repository_explorer import RepositoryExplorer


class _NoOptionalRetrievalRouter:
    pass


def test_repository_explorer_skips_missing_optional_router_capabilities(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    (java / "Network.java").write_text(
        "package example; public final class Network { public static void registerPacket(){ Registry.register(); } }\n",
        encoding="utf-8",
    )
    result = RepositoryExplorer(
        ProjectIndex(root),
        router=_NoOptionalRetrievalRouter(),
    ).explore("which API should register packet callback", line_budget=20)
    assert result.regions
    assert result.semantic_used is False
    assert result.rerank_used is False


def test_auto_test_time_scaling_respects_native_parallel_budget(monkeypatch) -> None:
    from minecraft_mod_ai import inference_time_scaling

    monkeypatch.setenv("MMM_TEST_TIME_SCALING", "auto")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert inference_time_scaling._scaling_mode() == "off"

    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert inference_time_scaling._scaling_mode() == "auto"

    monkeypatch.setenv("MMM_TEST_TIME_SCALING", "on")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert inference_time_scaling._scaling_mode() == "on"


def test_auto_repair_width_keeps_single_decode_when_one_slot(monkeypatch) -> None:
    from minecraft_mod_ai import agentic_optimization_contract as agentic

    monkeypatch.setenv("MMM_TEST_TIME_SCALING", "auto")
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    evidence = {
        "diagnostics": {},
        "build": {"status": "FAIL", "error": "x" * 200},
    }
    assert agentic._repair_candidate_count(engine, evidence, ()) == 1


def test_research_symbol_filter_drops_zero_score_global_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from minecraft_mod_ai import research_code_context as research

    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    (java / "Entry.java").write_text(
        """package example;
public final class Entry {
    public void tick() {
        Service.compute();
    }
}
""",
        encoding="utf-8",
    )
    (java / "Service.java").write_text(
        """package example;
public final class Service {
    public static void compute() {}

    public void unrelated() {}
}
""",
        encoding="utf-8",
    )

    class Router:
        def rerank(self, query, documents):
            return [1.0 if " tick" in (" " + document.casefold()) else 0.0 for document in documents]

    monkeypatch.setattr(
        research,
        "adapter_for_target",
        lambda _version, _loader: SimpleNamespace(
            loader="fabric",
            fabric_loader="test-loader",
            fabric_api="test-api",
            yarn_mappings="test",
            fabric_loom="test-loom",
        ),
    )
    module = SimpleNamespace(
        kind="custom_java",
        config={"feature": "tick"},
        depends_on=(),
        required_gates=(),
    )
    context = research.ResearchCodeContext(
        root,
        project_index=ProjectIndex(root),
        router=Router(),
        module=module,
        minecraft_version="test",
        loader="fabric",
        mappings="test",
        byte_budget=8192,
    )
    entries = context._entry_points("tick")
    assert entries
    assert {item.name for item in entries} == {"tick"}


def test_research_metric_vector_always_preserves_plan_alignment() -> None:
    from minecraft_mod_ai import research_code_context as research

    quality = research._quality(
        "public int compute(){ return normalize(1); }",
        path="src/main/java/example/Service.java",
    )
    metrics = research._retrieval_metrics(
        "Service compute dependency API validate",
        "public int compute(){ return normalize(1); }",
        path="src/main/java/example/Service.java",
        symbols=("compute",),
        graph_hop=1,
        quality=quality,
        target_plan="locate contract -> call normalize -> validate",
        example_plan="call normalize",
    )
    assert "plan_alignment" in metrics
    weights = research._adaptive_weights("Service.compute dependency API", metrics)
    assert "plan_alignment" in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_criterion_progress_path_copy_preserves_unrelated_payload_identity() -> None:
    from minecraft_mod_ai.planning_criterion_fragments import store_criterion_progress

    unrelated = {"rows": [object() for _ in range(32)]}
    state = {"large_unrelated_payload": unrelated, "detail_progress": []}
    result = store_criterion_progress(
        state,
        requirement_ref="REQ-1",
        selected_sections=("behavior",),
        criterion_index=0,
        criterion="works",
        fragment={"section_updates": []},
    )

    assert result is not state
    assert result["large_unrelated_payload"] is unrelated
    assert state["detail_progress"] == []
    assert len(result["detail_progress"]) == 1


def test_planner_checkpoint_does_not_clone_unrelated_nested_payload(monkeypatch) -> None:
    from minecraft_mod_ai import planning_state_adaptive_implementation as planning

    unrelated = {"large": [object() for _ in range(32)]}
    snapshots = []
    monkeypatch.setattr(planning, "_rehash", lambda value: value)
    monkeypatch.setattr(planning, "validate_planning_state", lambda value: None)

    state = {"large_unrelated_payload": unrelated, "state_sha256": "old"}
    result = planning._checkpoint_state(state, snapshots.append)

    assert result is not state
    assert result["large_unrelated_payload"] is unrelated
    assert snapshots[0] is not result
    assert snapshots[0]["large_unrelated_payload"] is unrelated


def test_artifact_progress_path_copy_preserves_unrelated_payload_identity() -> None:
    from minecraft_mod_ai import planning_state_adaptive_implementation as planning

    unrelated = {"rows": [object() for _ in range(32)]}
    state = {"large_unrelated_payload": unrelated, "artifact_progress": {}}
    result = planning._store_artifact_progress(
        state,
        requirement_ref="REQ-1",
        artifact_kind="item",
        step_id="feature/item/register",
        receipt={"status": "PLANNED"},
    )

    assert result is not state
    assert result["large_unrelated_payload"] is unrelated
    assert state["artifact_progress"] == {}
    assert result["artifact_progress"]["REQ-1"]["item"]["feature/item/register"] == {
        "status": "PLANNED"
    }


def test_completed_planning_job_releases_heavy_payload() -> None:
    import gc
    import weakref

    from minecraft_mod_ai import planning_state_adaptive_implementation as planning

    class Payload:
        pass

    payload = Payload()
    payload_ref = weakref.ref(payload)
    job = {
        "requirement_ref": "REQ-1",
        "evidence": payload,
        "fragments": {0: payload},
        "artifact_fragments": {"item": {"step": payload}},
    }

    planning._release_completed_job_payload(job)
    del payload
    gc.collect()

    assert job == {"requirement_ref": "REQ-1"}
    assert payload_ref() is None


def test_deadline_executor_streams_with_bounded_submission_window(monkeypatch) -> None:
    from concurrent.futures import Future

    from minecraft_mod_ai import deadline_executor as deadline

    instances = []
    consumed = []

    class ImmediateExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.shutdown_calls = []
            instances.append(self)

        def submit(self, function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, *, wait=True, cancel_futures=False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    def source():
        for value in range(5):
            consumed.append(value)
            yield value

    monkeypatch.setattr(deadline, "ThreadPoolExecutor", ImmediateExecutor)

    results = deadline.iter_completed_with_deadlines(
        source(),
        lambda value: value * 10,
        max_workers=2,
        stage="test-stage",
    )

    assert instances == []
    assert next(results) == (0, 0)
    assert consumed == [0, 1]
    assert instances[0].shutdown_calls == []

    results.close()
    assert instances[0].shutdown_calls == [(False, True)]


def test_deadline_executor_does_not_retain_completed_payload_history(monkeypatch) -> None:
    from concurrent.futures import Future
    import gc
    import weakref

    from minecraft_mod_ai import deadline_executor as deadline

    class Payload:
        def __init__(self, value: int) -> None:
            self.value = value

    class ImmediateExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.shutdown_calls = []

        def submit(self, function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, *, wait=True, cancel_futures=False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(deadline, "ThreadPoolExecutor", ImmediateExecutor)

    results = deadline.iter_completed_with_deadlines(
        [1, 2],
        Payload,
        max_workers=1,
        stage="memory-release-stage",
    )
    first = next(results)
    first_payload = weakref.ref(first[1])
    del first

    second = next(results)
    gc.collect()
    assert first_payload() is None
    assert second[0] == 2
    assert second[1].value == 2

    results.close()


def test_deadline_executor_timeout_cancels_before_nonblocking_shutdown(monkeypatch) -> None:
    from concurrent.futures import Future

    import pytest

    from minecraft_mod_ai import deadline_executor as deadline

    instances = []

    class NeverCompletingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.futures = []
            self.shutdown_calls = []
            instances.append(self)

        def submit(self, function, *args, **kwargs):
            future = Future()
            self.futures.append(future)
            return future

        def shutdown(self, *, wait=True, cancel_futures=False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(deadline, "ThreadPoolExecutor", NeverCompletingExecutor)
    monkeypatch.setattr(deadline, "planning_work_unit_timeout_seconds", lambda: 60.0)
    monkeypatch.setattr(
        deadline,
        "planning_stage_deadline",
        lambda *, work_units, workers, started_at: started_at,
    )

    with pytest.raises(deadline.ParallelExecutionTimeout):
        deadline.collect_completed_with_deadlines(
            ["blocked"],
            lambda item: item,
            max_workers=1,
            stage="timeout-stage",
        )

    assert instances
    assert all(future.cancelled() for future in instances[0].futures)
    assert instances[0].shutdown_calls == [(False, True)]
