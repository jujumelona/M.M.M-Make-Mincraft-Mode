from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from types import SimpleNamespace

from minecraft_mod_ai import design_record_runtime as design_runtime
from minecraft_mod_ai import feature_template_pipeline as feature_pipeline
from minecraft_mod_ai import prompt_template_pipeline as prompt_pipeline
from minecraft_mod_ai.parallel_model_tasks import (
    deterministic_model_map,
    serialized_callback,
)


class _Registry:
    @staticmethod
    def role(profile, role):
        del profile, role
        return SimpleNamespace(
            exclusive_gpu=True,
            provider="local",
            adapter="llama_cpp",
        )


class _Router:
    profile = "test"
    registry = _Registry()


def test_deterministic_model_map_uses_measured_slots_and_preserves_context(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _Router()
    marker = ContextVar("marker", default="missing")
    marker.set("root-context")
    lock = threading.Lock()
    active = 0
    maximum = 0

    def worker(value):
        nonlocal active, maximum
        assert marker.get() == "root-context"
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return value * 10

    assert deterministic_model_map(router, range(6), worker) == [0, 10, 20, 30, 40, 50]
    assert maximum == 2


def test_serialized_callback_keeps_parallel_state_writes_single_threaded():
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls = []

    def callback(value):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        calls.append(value)
        with lock:
            active -= 1

    safe = serialized_callback(callback)
    assert safe is not None
    threads = [threading.Thread(target=safe, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum == 1
    assert sorted(calls) == list(range(6))


def test_design_properties_use_native_parallel_slots_without_prompt_chaining(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _Router()
    lock = threading.Lock()
    active = 0
    maximum = 0
    accepted_snapshots: list[list[dict[str, object]]] = []

    context = {
        "allowed_properties": ["health", "speed", "attack_damage"],
        "required_properties": ["health", "speed", "attack_damage"],
    }

    monkeypatch.setattr(design_runtime, "load_record_template", lambda identifier: {"id": identifier})
    monkeypatch.setattr(
        design_runtime,
        "task_context",
        lambda template, supplied: dict(supplied),
    )

    def fake_single(router, identifier, *, context, **kwargs):
        nonlocal active, maximum
        del router, identifier, kwargs
        with lock:
            accepted_snapshots.append(list(context["accepted_records"]))
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {
            "property": context["requested_property"],
            "value": str(context["record_index"]),
        }

    monkeypatch.setattr(design_runtime, "run_single_record_template", fake_single)

    records, normalized = design_runtime._run_properties(
        router,
        "design/content_property",
        context,
        progress=None,
        checkpoint=None,
    )

    assert normalized == context
    assert [record["property"] for record in records] == context["required_properties"]
    assert [record["value"] for record in records] == ["0", "1", "2"]
    assert maximum == 2
    assert accepted_snapshots == [[], [], []]


def test_prompt_templates_execute_independently_and_merge_in_workflow_order(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _Router()
    lock = threading.Lock()
    active = 0
    maximum = 0

    def enter_model_call():
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    def fake_records(router, identifier, **kwargs):
        del router, kwargs
        enter_model_call()
        return {"records": [{"id": identifier}], "reason": "", "evidence_refs": []}

    def fake_value(router, identifier, **kwargs):
        del router
        if identifier != "prompt/capture":
            enter_model_call()
        if identifier == "prompt/intent":
            return {"statement": "goal"}
        if identifier == "prompt/scope":
            return {"scope_status": "explicit"}
        return kwargs["context"]

    monkeypatch.setattr(prompt_pipeline, "run_record_template", fake_records)
    monkeypatch.setattr(prompt_pipeline, "run_value_template", fake_value)

    result = prompt_pipeline.extract_prompt_records(router, "make a mod")
    assert result["goal"] == {"statement": "goal"}
    assert result["scope_status"] == "explicit"
    assert [row["id"] for row in result["known"]] == [
        "prompt/parse",
        "prompt/constraints",
        "prompt/output_requirements",
    ]
    assert result["references"] == [{"id": "prompt/entity_resolution"}]
    assert result["unresolved"] == [{"id": "prompt/ambiguities"}]
    assert maximum == 2


def test_feature_detail_dag_parallelizes_only_ready_steps(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _Router()
    lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_record(router, identifier, *, context, **kwargs):
        nonlocal active, maximum
        del router, kwargs
        step = identifier.split("/", 1)[1]
        assert set(context["relevant_sections"]) == set(
            feature_pipeline.FEATURE_CONTEXT_DEPENDENCIES[step]
        )
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"records": [{"step": step}], "reason": "", "evidence_refs": [step]}

    monkeypatch.setattr(feature_pipeline, "run_record_template", fake_record)
    sections, refs = feature_pipeline._run_feature_sections(
        router,
        {"feature_id": "f", "feature_description": "feature"},
        allowed_refs=set(),
    )
    assert tuple(sections) == feature_pipeline.FEATURE_DETAIL_STEPS
    assert refs == list(feature_pipeline.FEATURE_DETAIL_STEPS)
    assert maximum == 2


def test_atomic_checks_parallelize_after_sections_are_complete(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _Router()
    sections = {name: {"records": [], "not_applicable_reason": ""}
                for name in feature_pipeline.FEATURE_DETAIL_STEPS}
    lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_record(router, identifier, *, context, **kwargs):
        nonlocal active, maximum
        del router, kwargs
        assert identifier == "feature/atomic_check"
        check = context["target_check"]
        assert set(context["relevant_sections"]) == set(
            feature_pipeline.ATOMIC_CONTEXT_DEPENDENCIES[check]
        )
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {
            "records": [{"check": check, "passed": True, "reason": "ok"}],
            "reason": "",
            "evidence_refs": [],
        }

    monkeypatch.setattr(feature_pipeline, "run_record_template", fake_record)
    result = feature_pipeline._run_atomic_checks(
        router,
        {"feature_id": "f", "feature_description": "feature", "sections": sections},
        allowed_refs=set(),
    )
    assert result["atomic"] is True
    assert [row["check"] for row in result["checks"]] == list(feature_pipeline.ATOMIC_CHECKS)
    assert maximum == 2
