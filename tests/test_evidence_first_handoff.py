from __future__ import annotations

from copy import deepcopy

import pytest

import minecraft_mod_ai.evidence_first_handoff as handoff


def _plan() -> dict:
    return {
        "plan_sha256": "plan-sha",
        "reuse_decisions": [
            {
                "requirement_ref": "req-retain",
                "capability": "registry",
                "action": "retain",
                "component_refs": ["component-existing"],
                "source_refs": [],
            },
            {
                "requirement_ref": "req-fresh",
                "capability": "registry",
                "action": "fresh",
                "component_refs": [],
                "source_refs": [],
            },
            {
                "requirement_ref": "req-adapt",
                "capability": "resource",
                "action": "adapt",
                "component_refs": ["component-donor"],
                "source_refs": ["evidence-donor"],
            },
        ],
        "gap_catalog": [
            {"gap_id": "gap-fresh", "requirement_ref": "req-fresh"},
            {"gap_id": "gap-adapt", "requirement_ref": "req-adapt"},
        ],
        "tasks": [
            {
                "task_id": "task-fresh",
                "requirement_refs": ["req-fresh"],
                "gap_refs": ["gap-fresh"],
                "reuse_refs": [],
                "depends_on": [],
                "owned_anchors": [
                    {
                        "kind": "symbol",
                        "locator": "ExampleRegistry",
                        "module_id": "common",
                        "source_set": "main",
                    },
                    {
                        "kind": "resource",
                        "locator": "assets/example/models/item/example.json",
                        "module_id": "common",
                        "source_set": "main",
                    },
                ],
                "required_gates": ["compile"],
            },
            {
                "task_id": "task-adapt",
                "requirement_refs": ["req-adapt"],
                "gap_refs": ["gap-adapt"],
                "reuse_refs": ["component-donor", "evidence-donor"],
                "depends_on": ["task-fresh"],
                "owned_anchors": [
                    {
                        "kind": "loader_module",
                        "locator": "fabric",
                        "module_id": "fabric",
                        "source_set": "main",
                    }
                ],
                "required_gates": ["compile", "loader-smoke"],
            },
        ],
    }


@pytest.fixture(autouse=True)
def _isolate_handoff_from_planner_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handoff, "validate_evidence_first_plan", lambda _plan: None)


def test_handoff_suppresses_retain_and_preserves_exact_bindings() -> None:
    result = handoff.build_evidence_first_handoff(_plan())

    assert result["retain_receipts"] == [
        {
            "requirement_ref": "req-retain",
            "capability": "registry",
            "component_refs": ["component-existing"],
            "suppressed_task_generation": True,
        }
    ]
    assert result["work_graph"] == {
        "task_refs": ["task-fresh", "task-adapt"],
        "edges": [{"from_task_ref": "task-fresh", "to_task_ref": "task-adapt"}],
    }

    modules = {item["task_ref"]: item for item in result["production_modules"]}
    assert set(modules) == {"task-fresh", "task-adapt"}
    assert modules["task-fresh"]["requirement_refs"] == ["req-fresh"]
    assert modules["task-fresh"]["gap_refs"] == ["gap-fresh"]
    assert modules["task-fresh"]["reuse_action"] == "fresh"
    assert modules["task-fresh"]["reuse_refs"] == []
    assert modules["task-adapt"]["requirement_refs"] == ["req-adapt"]
    assert modules["task-adapt"]["gap_refs"] == ["gap-adapt"]
    assert modules["task-adapt"]["reuse_action"] == "adapt"
    assert modules["task-adapt"]["reuse_refs"] == ["component-donor", "evidence-donor"]

    assert result["asset_requests"] == [
        {
            "asset_request_id": result["asset_requests"][0]["asset_request_id"],
            "task_ref": "task-fresh",
            "requirement_refs": ["req-fresh"],
            "gap_refs": ["gap-fresh"],
            "reuse_action": "fresh",
            "reuse_refs": [],
            "locator": "assets/example/models/item/example.json",
            "module_id": "common",
            "source_set": "main",
            "required_gates": ["compile"],
        }
    ]


def test_handoff_is_deterministic() -> None:
    first = handoff.build_evidence_first_handoff(_plan())
    second = handoff.build_evidence_first_handoff(deepcopy(_plan()))

    assert first == second
    assert first["handoff_sha256"] == second["handoff_sha256"]


def test_handoff_rejects_retained_requirement_leaking_into_gap_work() -> None:
    plan = _plan()
    plan["gap_catalog"].append({"gap_id": "gap-retain", "requirement_ref": "req-retain"})

    with pytest.raises(handoff.EvidencePlanError, match="Retained requirement leaked"):
        handoff.build_evidence_first_handoff(plan)


def test_handoff_rejects_inexact_adaptation_refs() -> None:
    plan = _plan()
    plan["tasks"][1]["reuse_refs"] = ["component-donor"]

    with pytest.raises(handoff.EvidencePlanError, match="reuse refs do not exactly match"):
        handoff.build_evidence_first_handoff(plan)


def test_handoff_rejects_dangling_work_graph_dependency() -> None:
    plan = _plan()
    plan["tasks"][1]["depends_on"] = ["missing-task"]

    with pytest.raises(handoff.EvidencePlanError, match="invalid exact edge"):
        handoff.build_evidence_first_handoff(plan)
