from __future__ import annotations

import pytest

from minecraft_mod_ai.evidence_execution_contract import _execution_task
from minecraft_mod_ai.plan_collect_all_linker import (
    PlanCollectAllLinkError,
    validate_plan_collect_all,
)
from minecraft_mod_ai.task_execution_classification import claims_runtime, is_test_anchor


def _runtime_gametest_task(
    *,
    kind: str = "test",
    runtime_capability: bool = True,
) -> dict[str, object]:
    return {
        "task_id": "task_space_launch_prerequisite_gate_regression",
        "semantic_outcome": "Enforce the space launch prerequisite at runtime",
        "provides": (
            ["capability:space_launch_prerequisite"] if runtime_capability else []
        ),
        "required_gates": ["target_compile"],
        "requirement_refs": [],
        "gap_refs": [],
        "reuse_refs": [],
        "owned_anchors": [
            {
                "kind": kind,
                "locator": "src/gametest/java/generated/SpaceLaunchGameTest.java#SpaceLaunchGameTest",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "gametest",
            }
        ],
    }


def _minimal_plan_context() -> dict[str, object]:
    return {
        "ownership_context": {
            "source_root": "src/main/java",
            "namespace": "generated.generated_mod",
            "extension": "java",
            "module_id": ":",
            "source_set": "main",
        },
        "derived_requirement_ledger": {"facet_decisions": []},
    }


def test_runtime_semantic_without_capability_export_is_not_runtime() -> None:
    task = _runtime_gametest_task(runtime_capability=False)

    assert not claims_runtime(task)


def test_explicit_capability_export_is_runtime_authority() -> None:
    task = _runtime_gametest_task()

    assert claims_runtime(task)


def test_gametest_path_is_test_even_when_planner_labels_it_symbol() -> None:
    anchor = _runtime_gametest_task(kind="symbol")["owned_anchors"][0]

    assert isinstance(anchor, dict)
    assert is_test_anchor(anchor)


def test_runtime_gametest_only_task_is_lowered_to_production_and_passes_linker() -> None:
    lowered = _execution_task(_minimal_plan_context(), _runtime_gametest_task())

    assert lowered["execution_role"] == "production_with_verification"
    source_anchors = [
        anchor
        for anchor in lowered["owned_anchors"]
        if anchor.get("kind") == "symbol"
        and str(anchor.get("locator") or "").startswith("src/main/java/")
    ]
    assert len(source_anchors) == 1
    assert source_anchors[0]["status"] == "host_reserved"

    handoff = {
        "production_modules": [
            {
                "production_module_id": "regression-production-binding",
                "task_ref": lowered["task_id"],
                "module_id": source_anchors[0]["module_id"],
                "source_set": source_anchors[0]["source_set"],
                "reuse_action": "fresh",
                "owned_anchors": [source_anchors[0]],
            }
        ],
        "asset_requests": [],
    }

    validate_plan_collect_all({"tasks": [lowered]}, handoff)


def test_preflight_exception_contains_task_and_binding_diagnostics() -> None:
    task = _runtime_gametest_task()
    task["required_gates"] = []

    with pytest.raises(PlanCollectAllLinkError) as exc_info:
        validate_plan_collect_all(
            {"tasks": [task]},
            {"production_modules": [], "asset_requests": []},
        )

    message = str(exc_info.value)
    assert "TASK_EXECUTABLE_BINDING_MISSING" in message
    assert "TASK_RUNTIME_TEST_ONLY" in message
    assert "diagnostics=" in message
    assert "task_space_launch_prerequisite_gate_regression" in message
    assert "semantic_outcome" in message
    assert "owned_anchors" in message
    assert "production_bindings" in message
