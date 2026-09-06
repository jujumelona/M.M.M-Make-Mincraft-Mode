from __future__ import annotations

import pytest

from minecraft_mod_ai.coder_execution_contract import (
    build_coder_execution_contract,
    project_task_for_coder,
)


def _task() -> dict[str, object]:
    return {
        "task_id": "task_example",
        "task_sha256": "sha256:" + "1" * 64,
        "semantic_outcome": "Implement the approved example behavior.",
        "execution_role": "production",
        "requirement_refs": ["REQ-1"],
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "yarn",
            "java_version": "21",
        },
        "owned_anchors": [
            {"kind": "symbol", "locator": "src/main/java/X.java#X"}
        ],
        "reuse_refs": ["reuse:selected"],
        "depends_on": ["task_previous"],
        "consumes": ["capability:previous"],
        "provides": ["requirement_done:REQ-1"],
        "implementation_obligations": [
            "Implement exact behavior A.",
            "Implement exact behavior B.",
        ],
        "acceptance": ["Behavior A is observable.", "Behavior B is verified."],
        "required_gates": ["target_compile", "gametest"],
        "impact_probes": ["changed_symbols"],
        "original_prompt": "This must never reach the coder contract.",
        "engineering_worksheet": {"huge": "host-only"},
        "unrelated_requirement": {"REQ-2": "must not leak"},
    }


def test_projection_contains_only_task_local_execution_authority() -> None:
    projected = project_task_for_coder(_task())
    contract = projected["coder_execution_contract"]

    assert set(projected) == {"task_id", "task_sha256", "coder_execution_contract"}
    assert contract["task_ref"] == "task_example"
    assert contract["requirement_refs"] == ["REQ-1"]
    assert contract["implementation_steps"] == [
        {"step": 1, "obligation": "Implement exact behavior A."},
        {"step": 2, "obligation": "Implement exact behavior B."},
    ]
    assert contract["acceptance_checks"] == [
        "Behavior A is observable.",
        "Behavior B is verified.",
    ]
    rendered = repr(projected)
    assert "original_prompt" not in rendered
    assert "engineering_worksheet" not in rendered
    assert "REQ-2" not in rendered
    assert "unrelated_requirement" not in rendered


def test_empty_obligation_list_uses_host_owned_semantic_outcome_once() -> None:
    task = _task()
    task["implementation_obligations"] = []

    contract = build_coder_execution_contract(task)

    assert contract["implementation_steps"] == [
        {"step": 1, "obligation": "Implement the approved example behavior."}
    ]


def test_contract_is_deterministic_and_hash_bound() -> None:
    first = build_coder_execution_contract(_task())
    second = build_coder_execution_contract(dict(reversed(list(_task().items()))))

    assert first == second
    assert first["contract_sha256"].startswith("sha256:")
    assert len(first["contract_sha256"]) == 71


@pytest.mark.parametrize(
    "field",
    ["task_id", "semantic_outcome"],
)
def test_missing_execution_identity_fails_closed(field: str) -> None:
    task = _task()
    task[field] = ""

    with pytest.raises(ValueError, match="CODER_EXECUTION_CONTRACT"):
        build_coder_execution_contract(task)
