from __future__ import annotations

import pytest

from minecraft_mod_ai.coder_execution_contract import (
    CODER_EXECUTION_CONTRACT_SCHEMA,
    build_coder_execution_contract,
    project_task_for_coder,
)
from minecraft_mod_ai.implementation_template_contract import (
    SCHEMA,
    build_implementation_template,
)


def _task() -> dict[str, object]:
    return {
        "task_id": "task_example",
        "task_sha256": "sha256:" + "1" * 64,
        "sequence": 1,
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
            {
                "kind": "symbol",
                "locator": "src/main/java/X.java#X",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            }
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
        "original_prompt": "This must never reach the coder envelope.",
        "unrelated_requirement": {"REQ-2": "must not leak"},
    }


def test_compatibility_surface_uses_one_canonical_contract_owner() -> None:
    task = _task()
    compatibility = build_coder_execution_contract(task)
    canonical = build_implementation_template(task)

    assert CODER_EXECUTION_CONTRACT_SCHEMA == SCHEMA == "mmm/coder-execution-contract"
    assert compatibility == canonical
    assert compatibility["task_ref"] == "task_example"
    assert [step["obligation"] for step in compatibility["implementation_steps"]] == [
        "Implement exact behavior A.",
        "Implement exact behavior B.",
    ]
    assert compatibility["dataflow"] == {
        "consumes": ["capability:previous"],
        "provides": ["requirement_done:REQ-1"],
    }


def test_projection_contains_only_identity_and_canonical_contract() -> None:
    projected = project_task_for_coder(_task())

    assert set(projected) == {"task_id", "task_sha256", "coder_execution_contract"}
    assert projected["task_id"] == "task_example"
    rendered = repr(projected)
    assert "original_prompt" not in rendered
    assert "REQ-2" not in rendered
    assert "unrelated_requirement" not in rendered


def test_contract_is_deterministic_and_hash_bound() -> None:
    first = build_coder_execution_contract(_task())
    second = build_coder_execution_contract(dict(reversed(list(_task().items()))))

    assert first == second
    assert first["contract_sha256"].startswith("sha256:")
    assert len(first["contract_sha256"]) == 71


def test_compatibility_builder_does_not_restore_semantic_only_fallback() -> None:
    task = _task()
    task["implementation_obligations"] = []

    with pytest.raises(ValueError, match="semantic-only"):
        build_coder_execution_contract(task)
