from __future__ import annotations

import pytest

from minecraft_mod_ai.atomic_concern_source import (
    _structure_scan,
    _type_leaf_names,
    _validate_declared_type_authority,
    _validate_first_pass_java_semantics,
)
from minecraft_mod_ai.production_local_type_recovery import (
    canonicalize_plan_local_zero_arg_domain_types,
    region_correction_for_rejection,
    type_authority_repair_contract,
)
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.execution_contract_policy import (
    java_generation_recipe_contract,
    java_region_system_prompt_contract,
)


def _authority() -> dict[str, object]:
    return {
        "source_requirements": {
            "R1": (
                "- responsibilities: caller callee contract: player calls "
                "build_ship(), system returns ShipObject."
            )
        }
    }


def test_production_materializes_requirement_owned_zero_arg_domain_type() -> None:
    source = (
        "private static ShipObject buildShip() {\n"
        "    return new ShipObject();\n"
        "}"
    )

    repaired, changes = canonicalize_plan_local_zero_arg_domain_types(
        source,
        concern_authority=_authority(),
        dependency_source="",
        sibling_api=(),
        validate_declared_type_authority=_validate_declared_type_authority,
        type_leaf_names=_type_leaf_names,
        structure_scan=_structure_scan,
    )

    assert "private static final class ShipObject {}" in repaired
    assert changes == ("ShipObject:materialized_private_zero_arg_domain_type",)
    _validate_first_pass_java_semantics(
        repaired,
        dependency_source="",
        sibling_api=(),
        validate_declared_type_authority=_validate_declared_type_authority,
        type_leaf_names=_type_leaf_names,
        structure_scan=_structure_scan,
    )


def test_production_does_not_hide_unowned_type_typo() -> None:
    source = (
        "private static ReentrantLokk buildShip() {\n"
        "    return new ReentrantLokk();\n"
        "}"
    )

    repaired, changes = canonicalize_plan_local_zero_arg_domain_types(
        source,
        concern_authority=_authority(),
        dependency_source="",
        sibling_api=(),
        validate_declared_type_authority=_validate_declared_type_authority,
        type_leaf_names=_type_leaf_names,
        structure_scan=_structure_scan,
    )

    assert repaired == source
    assert changes == ()
    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*ReentrantLokk",
    ):
        _validate_first_pass_java_semantics(
            repaired,
            dependency_source="",
            sibling_api=(),
        )


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


def test_small_model_prompt_explicitly_forbids_undeclared_simple_domain_types() -> None:
    recipe = java_generation_recipe_contract("responsibilities")
    rule = recipe["declare_plan_local_domain_type_rule"]

    assert "Never emit an undeclared simple Java type" in rule
    assert "private static nested class/record" in rule

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
        "or the exact fully-qualified external type."
    )

    assert contract is not None
    assert contract["unknown_simple_types"] == ["ShipObject"]
    assert "declare the smallest private static nested class/record" in contract["rules"]
