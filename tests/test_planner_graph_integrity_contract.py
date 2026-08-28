from __future__ import annotations

import inspect
from types import SimpleNamespace

from minecraft_mod_ai import planner_graph_integrity_contract as contract


def test_contract_does_not_own_semantic_interpretation_or_catalog_construction() -> None:
    source = inspect.getsource(contract)

    assert "semantic_requirement_authority" in source  # documented ownership boundary
    assert "_semantic._call_semantic_model" not in source
    assert "_semantic._generate_approved_nodes" not in source
    assert "_semantic._build_catalog" not in source
    assert "_stub_semantic_model =" not in source
    assert "_invoke_semantic_model =" not in source
    assert "build_authoritative_request_catalog =" not in source
    assert source.count("_evidence._compile_tasks =") == 1


def test_cross_system_dependencies_are_promoted_into_task_dag(monkeypatch):
    definitions = {
        "feature.a": SimpleNamespace(default_dependencies=("feature.b",)),
        "feature.b": SimpleNamespace(default_dependencies=("feature.c",)),
        "feature.c": SimpleNamespace(default_dependencies=()),
    }
    monkeypatch.setattr(contract, "atomic_capability_definitions", lambda: definitions)

    gaps = (
        {
            "gap_id": "gap_a",
            "capability": "feature.a",
            "missing_provides": ["capability:feature.a"],
        },
        {
            "gap_id": "gap_b",
            "capability": "feature.b",
            "missing_provides": ["capability:feature.b"],
        },
        {
            "gap_id": "gap_c",
            "capability": "feature.c",
            "missing_provides": ["capability:feature.c"],
        },
    )

    def fake_compile(_gaps, _reuse, _target, _branches, _ownership):
        return (
            {
                "task_id": "task_a",
                "gap_refs": ["gap_a"],
                "consumes": ["target:frozen"],
                "provides": ["capability:feature.a"],
                "depends_on": [],
                "task_sha256": "",
            },
            {
                "task_id": "task_b",
                "gap_refs": ["gap_b"],
                "consumes": ["target:frozen"],
                "provides": ["capability:feature.b"],
                "depends_on": [],
                "task_sha256": "",
            },
            {
                "task_id": "task_c",
                "gap_refs": ["gap_c"],
                "consumes": ["target:frozen"],
                "provides": ["capability:feature.c"],
                "depends_on": [],
                "task_sha256": "",
            },
        )

    monkeypatch.setattr(
        contract._compile_tasks_with_cross_system_dependencies,
        "__wrapped__",
        fake_compile,
    )

    tasks = contract._compile_tasks_with_cross_system_dependencies(gaps, (), {}, {}, {})
    by_id = {item["task_id"]: item for item in tasks}

    assert "capability:feature.b" in by_id["task_a"]["consumes"]
    assert "capability:feature.c" in by_id["task_b"]["consumes"]
    assert by_id["task_a"]["depends_on"] == ["task_b"]
    assert by_id["task_b"]["depends_on"] == ["task_c"]
    assert by_id["task_c"]["depends_on"] == []
