from __future__ import annotations

from minecraft_mod_ai.evidence_execution_contract import _execution_task


def _plan() -> dict[str, object]:
    return {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "REQ-1",
                    "implementation_obligations": [
                        "Implement the exact REQ-1 behavior.",
                        "Preserve the REQ-1 invariant.",
                    ],
                    "acceptance": ["REQ-1 behavior is observable and verified."],
                },
                {
                    "requirement_id": "REQ-2",
                    "implementation_obligations": ["Implement unrelated REQ-2 behavior."],
                    "acceptance": ["REQ-2 passes."],
                },
            ]
        },
        "derived_requirement_ledger": {
            "facet_decisions": [
                {
                    "parent_requirement_ref": "REQ-1",
                    "disposition": "derived",
                    "owner_task_ref": "task-final",
                    "implementation_obligations": [
                        "Preserve the REQ-1 invariant.",
                        "Implement the grounded derived facet.",
                    ],
                    "acceptance": ["Derived facet is verified."],
                }
            ]
        },
        "ownership_context": {
            "source_root": "src/main/java",
            "namespace": "generated.example",
            "module_id": ":",
            "source_set": "main",
        },
    }


def _task(*, task_id: str, provides: list[str]) -> dict[str, object]:
    return {
        "task_id": task_id,
        "semantic_outcome": "resource",
        "requirement_refs": ["REQ-1"],
        "provides": provides,
        "required_gates": [],
        "owned_anchors": [
            {
                "kind": "resource",
                "locator": "src/main/resources/example.json",
                "ownership": "exclusive",
            }
        ],
        "acceptance": ["Task-local acceptance."],
    }


def test_completion_task_receives_only_its_planner_obligations_and_validation() -> None:
    lowered = _execution_task(
        _plan(),
        _task(task_id="task-final", provides=["requirement_done:REQ-1"]),
    )

    assert lowered["implementation_obligations"] == [
        "Implement the exact REQ-1 behavior.",
        "Preserve the REQ-1 invariant.",
        "Implement the grounded derived facet.",
    ]
    assert lowered["acceptance"] == [
        "Task-local acceptance.",
        "REQ-1 behavior is observable and verified.",
        "Derived facet is verified.",
    ]
    assert "Implement unrelated REQ-2 behavior." not in lowered["implementation_obligations"]
    assert "REQ-2 passes." not in lowered["acceptance"]


def test_intermediate_task_does_not_inherit_requirement_wide_obligations() -> None:
    lowered = _execution_task(
        _plan(),
        _task(task_id="task-intermediate", provides=["capability:partial"]),
    )

    assert lowered["implementation_obligations"] == []
    assert lowered["acceptance"] == ["Task-local acceptance."]


def test_completion_token_for_other_requirement_does_not_grant_req1_contract() -> None:
    lowered = _execution_task(
        _plan(),
        _task(task_id="task-intermediate", provides=["requirement_done:REQ-2"]),
    )

    assert lowered["implementation_obligations"] == []
    assert lowered["acceptance"] == ["Task-local acceptance."]
