from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import planner_graph_integrity_contract as contract


class _TextRouter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate_text(self, *args, **kwargs):
        self.calls += 1
        assert kwargs["response_format"] == "text"
        assert kwargs["response_schema"] is None
        assert kwargs["enable_tools"] is False
        return self.text

    def generate_tool_decision(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("semantic leaf planning must not use structured JSON/tool output")


def test_plain_semantic_leaf_protocol_has_no_small_plan_cardinality_cap():
    lines = [
        "\t".join(
            (
                "REQ",
                "0",
                f"feature.{index}",
                f"feature {index}",
                f"player-visible behavior {index}",
                f"given {index}",
                f"when {index}",
                f"then {index}",
            )
        )
        for index in range(12)
    ]
    router = _TextRouter("\n".join(lines))

    payload = contract._plain_semantic_call(
        router,
        ({"clause_index": 0, "text": "one authored clause with many independent mechanics"},),
    )

    assert router.calls == 1
    requirements = payload["requirements"]
    assert len(requirements) == 12
    assert [item["capability_id"] for item in requirements] == [
        f"feature.{index}" for index in range(12)
    ]


def test_legacy_semantic_fallback_preserves_all_host_recognized_candidates(monkeypatch):
    resolution = SimpleNamespace(
        nodes=[
            SimpleNamespace(capability_id=f"cap.{index}", origin="explicit")
            for index in range(12)
        ]
    )
    monkeypatch.setattr(
        contract,
        "resolve_capabilities_from_phrase_structured",
        lambda _text: resolution,
    )

    semantic = contract._host_stub_all_candidates("x", 0, 1, "x")

    assert semantic.gameplay_capability_candidates == tuple(
        f"cap.{index}" for index in range(12)
    )


def test_requirement_dependency_closure_recurses_until_fixed_point(monkeypatch):
    definitions = {
        f"cap.{index}": SimpleNamespace(
            description=f"capability {index}",
            default_dependencies=(f"cap.{index + 1}",) if index < 11 else (),
        )
        for index in range(12)
    }
    monkeypatch.setattr(contract, "atomic_capability_definitions", lambda: definitions)

    catalog = {
        "prompt_sha256": "p" * 64,
        "requirements": [
            {
                "requirement_id": "req_root",
                "capability": "cap.0",
                "statement": "root authored behavior",
                "semantic_statement": "root authored behavior",
                "mandatory": True,
                "provenance_role": "explicit",
                "source_span": {"start": 0, "end": 4, "quote": "root"},
                "derived_from": [],
                "depends_on": [],
                "provides": ["capability:cap.0"],
                "gameplay_capabilities": ["cap.0"],
                "implementation_capabilities": [],
                "artifact_task_ids": [],
                "semantic_status": "RESOLVED",
                "unresolved_spans": [],
                "acceptance": ["root is observable"],
                "observable_behavior": {"given": "g", "when": "w", "then": "t"},
            }
        ],
        "requirement_graph": {"node_ids": ["req_root"], "edges": []},
        "semantic_audit": {},
        "catalog_sha256": "",
    }

    expanded = contract._expand_catalog_dependency_closure(catalog)

    assert len(expanded["requirements"]) == 12
    assert {item["capability"] for item in expanded["requirements"]} == {
        f"cap.{index}" for index in range(12)
    }
    assert len(expanded["requirement_graph"]["edges"]) == 11
    assert expanded["semantic_audit"]["graph_cardinality_policy"] == (
        "unbounded_by_semantic_item_count"
    )
    assert expanded["semantic_audit"]["dependency_expansion"] == (
        "recursive_ontology_closure_until_fixed_point"
    )


def test_cross_system_dependencies_are_promoted_into_task_dag(monkeypatch):
    definitions = {
        "feature.a": SimpleNamespace(default_dependencies=("feature.b",)),
        "feature.b": SimpleNamespace(default_dependencies=("feature.c",)),
        "feature.c": SimpleNamespace(default_dependencies=()),
    }
    monkeypatch.setattr(contract, "atomic_capability_definitions", lambda: definitions)

    gaps = (
        {"gap_id": "gap_a", "capability": "feature.a", "missing_provides": ["capability:feature.a"]},
        {"gap_id": "gap_b", "capability": "feature.b", "missing_provides": ["capability:feature.b"]},
        {"gap_id": "gap_c", "capability": "feature.c", "missing_provides": ["capability:feature.c"]},
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
