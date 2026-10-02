from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.atomic_concern_source import (
    AtomicConcernExecutor,
    _validate_first_pass_java_semantics,
)
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.execution_contract_policy import (
    java_generation_recipe_contract,
    java_region_system_prompt_contract,
)
from minecraft_mod_ai.production_local_type_recovery import (
    region_correction_for_rejection,
    type_authority_repair_contract,
)


def _task() -> dict[str, object]:
    return {
        "task_id": "t",
        "semantic_outcome": "build a ship",
        "authored_atomic_contract": {
            "schema_version": "mmm/authored-atomic-contract-v1",
            "concerns": {
                "responsibilities": {
                    "source_requirements": {
                        "R1": "- responsibilities:",
                        "R2": (
                            "  - caller callee contract: player calls build_ship(), "
                            "system returns ShipObject."
                        )
                    },
                    "structured_records": [],
                }
            },
        },
    }


def _executor(
    outputs: list[str],
    captured: list[list[dict[str, str]]],
    *,
    attempt_limit: int = 2,
) -> AtomicConcernExecutor:
    remaining = list(outputs)

    def call_coder(messages):
        captured.append([dict(item) for item in messages])
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    return AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task=_task(),
        section="integration",
        concerns=(
            {
                "sequence": 0,
                "identifier": "responsibilities",
                "concern": "responsibilities",
                "task": "implement authored responsibility",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
        region_attempt_limit=attempt_limit,
        retry_structural_rejections=True,
    )


def test_ungrounded_plan_local_type_forces_model_concern_regeneration() -> None:
    captured: list[list[dict[str, str]]] = []
    executor = _executor(
        [
            (
                "private static ShipObject buildShip() {\n"
                "    return new ShipObject();\n"
                "}"
            ),
            (
                "private static final class ShipObject {}\n\n"
                "private static ShipObject buildShip() {\n"
                "    return new ShipObject();\n"
                "}"
            ),
        ],
        captured,
    )

    result = executor.run()

    assert len(captured) == 2
    retry_payload = json.loads(captured[1][-1]["content"])
    repair = retry_payload["type_authority_repair_contract"]
    assert repair["unknown_simple_types"] == ["ShipObject"]
    assert repair["mode"] == "mechanical_copy_edit"
    assert repair["mechanical_edits"] == [
        {
            "operation": "insert_exact_sibling_declaration",
            "type_name": "ShipObject",
            "exact_declaration": "private static final class ShipObject {}",
            "preserve_existing_declarations": [
                "private static ShipObject buildShip() { ... }"
            ],
            "edit_budget": "one exact sibling declaration insertion; no semantic rewrites",
        }
    ]
    assert "current_selected_region_source" in retry_payload
    assert "private static ShipObject buildShip()" in retry_payload[
        "current_selected_region_source"
    ]
    assert "region_correction" not in retry_payload
    assert "private static final class ShipObject {}" in result["source"]
    assert "private static ShipObject buildShip()" in result["source"]


def test_original_log_identical_retry_gets_one_more_mechanical_copy_edit_turn() -> None:
    captured: list[list[dict[str, str]]] = []
    bad = (
        "private static ShipObject buildShip() {\n"
        "    return new ShipObject();\n"
        "}"
    )
    executor = _executor(
        [
            bad,
            bad,
            (
                "private static final class ShipObject {}\n\n"
                "private static ShipObject buildShip() {\n"
                "    return new ShipObject();\n"
                "}"
            ),
        ],
        captured,
        attempt_limit=3,
    )

    result = executor.run()

    assert len(captured) == 3
    second = json.loads(captured[1][-1]["content"])[
        "type_authority_repair_contract"
    ]
    third = json.loads(captured[2][-1]["content"])[
        "type_authority_repair_contract"
    ]
    assert second["mode"] == "mechanical_copy_edit"
    assert second["retry_level"] == 1
    assert third["mode"] == "mechanical_copy_edit"
    assert third["retry_level"] == 2
    assert third["mechanical_edits"][0]["exact_declaration"] == (
        "private static final class ShipObject {}"
    )
    assert "private static final class ShipObject {}" in result["source"]


def test_bad_type_is_not_silently_materialized_by_host() -> None:
    captured: list[list[dict[str, str]]] = []
    executor = _executor(
        [
            (
                "private static ShipObject buildShip() {\n"
                "    return new ShipObject();\n"
                "}"
            )
        ],
        captured,
        attempt_limit=1,
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*ShipObject",
    ):
        executor.run()

    assert len(captured) == 1


def test_type_authority_failure_uses_full_concern_regeneration() -> None:
    diagnostic = (
        "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
        "ShipObject. Use an authoritative sibling/dependency type, a known JDK type, "
        "or the exact fully-qualified external type."
    )
    repair = type_authority_repair_contract(diagnostic)
    correction = region_correction_for_rejection(
        "private static ShipObject buildShip() { return null; }",
        diagnostic,
        allow_private_restructure=True,
        type_authority_repair=repair,
    )

    assert repair is not None
    assert correction is None


def test_non_type_failure_keeps_region_correction_shape_guard() -> None:
    correction = region_correction_for_rejection(
        "private static int value() { return 1; }",
        "ATOMIC_CONCERN_RESPONSE_INVALID: some other semantic problem",
        allow_private_restructure=True,
        type_authority_repair=None,
    )

    assert correction is not None


def test_unowned_typo_is_still_rejected() -> None:
    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*ReentrantLokk",
    ):
        _validate_first_pass_java_semantics(
            "private static ReentrantLokk lock;",
            dependency_source="",
            sibling_api=(),
        )


def test_small_model_prompt_forbids_undeclared_simple_domain_types() -> None:
    recipe = java_generation_recipe_contract("responsibilities")
    rule = recipe["declare_plan_local_domain_type_rule"]

    assert "Never emit an undeclared simple Java type" in rule
    assert "private static nested class/record" in rule
    assert "regenerate the whole selected concern" in rule

    system = java_region_system_prompt_contract(
        section="integration",
        concern_name="responsibilities",
        response_region="members",
        platform_api_policy="host_grounded_only",
    )
    assert "Never emit an undeclared simple Java type" in system
    assert "declare the smallest private static nested class/record" in system


def test_type_authority_repair_contract_names_exact_unknown_type() -> None:
    contract = type_authority_repair_contract(
        "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
        "ShipObject. Use an authoritative sibling/dependency type, a known JDK type, "
        "or the exact fully-qualified external type.",
        rejected_source=(
            "private static ShipObject buildShip() { return new ShipObject(); }"
        ),
        concern_authority=_task()["authored_atomic_contract"]["concerns"][
            "responsibilities"
        ],
    )

    assert contract is not None
    assert contract["unknown_simple_types"] == ["ShipObject"]
    assert contract["mode"] == "mechanical_copy_edit"
    assert contract["mechanical_edits"][0]["exact_declaration"] == (
        "private static final class ShipObject {}"
    )
    assert "copy-edit task" in contract["rules"]
