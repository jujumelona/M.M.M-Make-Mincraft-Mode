from __future__ import annotations

import pytest

from minecraft_mod_ai.atomic_concern_source import (
    _canonicalize_plan_local_zero_arg_domain_types,
    _type_authority_repair_contract,
    _validate_first_pass_java_semantics,
)
from minecraft_mod_ai.atomic_region_correction import RegionCorrection
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

    repaired, changes = _canonicalize_plan_local_zero_arg_domain_types(
        source,
        concern_authority=_authority(),
        dependency_source="",
        sibling_api=(),
    )

    assert "private static final class ShipObject {}" in repaired
    assert changes == ("ShipObject:materialized_private_zero_arg_domain_type",)
    _validate_first_pass_java_semantics(
        repaired,
        dependency_source="",
        sibling_api=(),
    )


def test_production_does_not_hide_unowned_type_typo() -> None:
    source = (
        "private static ReentrantLokk buildShip() {\n"
        "    return new ReentrantLokk();\n"
        "}"
    )

    repaired, changes = _canonicalize_plan_local_zero_arg_domain_types(
        source,
        concern_authority=_authority(),
        dependency_source="",
        sibling_api=(),
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


def test_type_authority_repair_can_add_exact_private_nested_type() -> None:
    rejected = "private static ShipObject buildShip() { return null; }"
    diagnostic = (
        "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
        "ShipObject. Use an authoritative sibling/dependency type, a known JDK type, "
        "or the exact fully-qualified external type."
    )
    correction = RegionCorrection.for_diagnostic(
        rejected,
        diagnostic,
        allow_private_restructure=True,
        allow_private_type_additions=True,
    )
    assert correction is not None

    merged = correction.merge(
        "private static final class ShipObject {}\n"
        "private static ShipObject buildShip() { return new ShipObject(); }"
    )

    assert "class ShipObject" in merged
    assert "buildShip()" in merged
    _validate_first_pass_java_semantics(
        merged,
        dependency_source="",
        sibling_api=(),
    )


def test_private_nested_type_addition_stays_forbidden_without_type_repair_authority() -> None:
    rejected = "private static ShipObject buildShip() { return null; }"
    correction = RegionCorrection.for_diagnostic(
        rejected,
        "ATOMIC_CONCERN_RESPONSE_INVALID: some other semantic problem",
        allow_private_restructure=True,
        allow_private_type_additions=False,
    )
    assert correction is not None

    with pytest.raises(
        CustomModuleGenerationError,
        match="new declaration outside private implementation",
    ):
        correction.merge(
            "private static final class ShipObject {}\n"
            "private static ShipObject buildShip() { return new ShipObject(); }"
        )


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
    contract = _type_authority_repair_contract(
        "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
        "ShipObject. Use an authoritative sibling/dependency type, a known JDK type, "
        "or the exact fully-qualified external type."
    )

    assert contract is not None
    assert contract["unknown_simple_types"] == ["ShipObject"]
    assert "declare the smallest private static nested class/record" in contract["rules"]
