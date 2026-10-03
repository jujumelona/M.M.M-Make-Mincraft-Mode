from __future__ import annotations

import copy

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.production_state_compiler import (
    compile_production_state_section,
    normalize_structured_state_section,
    render_production_state_java,
)
from minecraft_mod_ai.structured_state_runtime import validate_state_expression


def _state_section(
    *,
    variables=None,
    transitions=None,
    invariants=None,
    initialization=None,
    updates=None,
    cleanup=None,
    concurrency=None,
):
    return {
        "specification": {
            "variables": list(variables or []),
            "transitions": list(transitions or []),
            "invariants": list(invariants or []),
            "initialization": list(initialization or []),
            "updates": list(updates or []),
            "cleanup": list(cleanup or []),
            "concurrency": list(concurrency or []),
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }


def _plan(section=None) -> AuthoredPlan:
    plan = AuthoredPlan.__new__(AuthoredPlan)
    object.__setattr__(plan, "requested_prompt", "state compiler test")
    object.__setattr__(plan, "text", "# state_model")
    object.__setattr__(plan, "existing_input_sha256", "")
    object.__setattr__(plan, "media_paths", ())
    object.__setattr__(plan, "schema_version", "mmm/authored-plan-v2")
    object.__setattr__(
        plan,
        "structured_sections",
        {} if section is None else {"state_model": section},
    )
    object.__setattr__(plan, "typed_plan_ir", {})
    return plan


def _credits_section() -> dict:
    return _state_section(
        variables=[
            {
                "name": "credits",
                "owner": "player",
                "type": "integer",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            }
        ],
        invariants=[
            {
                "condition": "credits >= 0",
                "enforcement": "reject negative balances",
            }
        ],
        initialization=[
            {
                "state": "credits",
                "value": "0",
                "when": "player first joins",
            }
        ],
        updates=[
            {
                "trigger": "reward",
                "mutation": "credits = credits + 1",
                "postcondition": "credits >= 1",
            }
        ],
    )


def test_missing_state_section_returns_empty_sidecar() -> None:
    assert compile_production_state_section(_plan()) == {}


def test_compile_uses_structured_authority_only_and_does_not_mutate_plan() -> None:
    raw = _credits_section()
    plan = _plan(raw)
    before = copy.deepcopy(plan.structured_sections)

    compiled = compile_production_state_section(plan)

    assert compiled["specification"]["variables"][0]["name"] == "credits"
    assert compiled["specification"]["invariants"][0]["condition"] == "credits >= 0"
    assert plan.structured_sections == before


def test_normalization_accepts_section_body_or_specification_wrapper() -> None:
    wrapped = _credits_section()
    direct = wrapped["specification"]

    assert normalize_structured_state_section(wrapped)["specification"] == (
        normalize_structured_state_section(direct)["specification"]
    )


def test_state_variable_spellings_resolve_to_declared_identity() -> None:
    section = _state_section(
        variables=[
            {
                "name": "Ship Status",
                "owner": "ship",
                "type": "string",
                "unit": "status",
                "default": "Docked",
                "domain": "status",
            }
        ],
        invariants=[
            {
                "condition": "ship_status == Ready",
                "enforcement": "reject invalid launch",
            }
        ],
    )

    normalized = normalize_structured_state_section(section)
    invariant = normalized["specification"]["invariants"][0]

    assert "Ship_Status" in invariant["condition"] or "ship_status" in invariant["condition"]
    validate_state_expression(invariant["condition"])


def test_word_logic_is_normalized_to_host_expression_syntax() -> None:
    section = _state_section(
        variables=[
            {
                "name": "credits",
                "owner": "player",
                "type": "integer",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            },
            {
                "name": "level",
                "owner": "player",
                "type": "integer",
                "unit": "level",
                "default": "1",
                "domain": "integer >= 1",
            },
        ],
        invariants=[
            {
                "condition": "credits >= 0 and level >= 1",
                "enforcement": "reject invalid state",
            }
        ],
    )

    normalized = normalize_structured_state_section(section)
    condition = normalized["specification"]["invariants"][0]["condition"]

    assert "&&" in condition
    validate_state_expression(condition)


def test_external_subsystem_mutation_is_removed_instead_of_inventing_state() -> None:
    section = _state_section(
        variables=[
            {
                "name": "ship_state",
                "owner": "ship",
                "type": "string",
                "unit": "status",
                "default": "Docked",
                "domain": "status",
            }
        ],
        transitions=[
            {
                "from_state": "dock",
                "trigger": "launch",
                "guard": "ship_state == Ready",
                "mutation": "SendPacket",
                "to_state": "space",
            }
        ],
    )

    normalized = normalize_structured_state_section(section)
    transition = normalized["specification"]["transitions"][0]

    assert transition["mutation"] == ""
    assert all(
        row["name"] != "SendPacket"
        for row in normalized["specification"]["variables"]
    )


def test_semicolon_inside_quoted_value_is_not_split_as_multiple_mutations() -> None:
    section = _state_section(
        variables=[
            {
                "name": "label",
                "owner": "player",
                "type": "string",
                "unit": "text",
                "default": "",
                "domain": "text",
            }
        ],
        updates=[
            {
                "trigger": "rename",
                "mutation": 'label = "alpha;beta"',
                "postcondition": 'label == "alpha;beta"',
            }
        ],
    )

    normalized = normalize_structured_state_section(section)
    update = normalized["specification"]["updates"][0]

    assert "alpha;beta" in update["mutation"]
    assert len(normalized["specification"]["updates"]) == 1


def test_unknown_function_semantics_fail_closed() -> None:
    section = _state_section(
        variables=[
            {
                "name": "credits",
                "owner": "player",
                "type": "integer",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            }
        ],
        invariants=[
            {
                "condition": "mystery(credits) > 0",
                "enforcement": "reject invalid state",
            }
        ],
    )

    with pytest.raises(ValueError):
        normalize_structured_state_section(section)


def test_empty_canonical_state_cannot_render_java_owner() -> None:
    with pytest.raises(
        ValueError,
        match="PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED",
    ):
        render_production_state_java(
            _state_section(),
            package_name="ai.minecraft.state",
        )


def test_rendered_state_java_is_complete_host_owned_class() -> None:
    source = render_production_state_java(
        _credits_section(),
        package_name="ai.minecraft.state",
    )

    assert source.startswith("package ai.minecraft.state;")
    assert "// MMM:TYPED_PLAN_STATE_OWNER" in source
    assert "public final class AuthoredStateModel" in source
    assert "private AuthoredStateModel()" in source
    assert "credits" in source


def test_invalid_java_symbol_or_package_fails_closed() -> None:
    with pytest.raises(ValueError, match="PRODUCTION_STATE_SYMBOL_INVALID"):
        render_production_state_java(
            _credits_section(),
            package_name="ai.minecraft.state",
            symbol="bad symbol",
        )

    with pytest.raises(ValueError, match="PRODUCTION_STATE_PACKAGE_INVALID"):
        render_production_state_java(
            _credits_section(),
            package_name="bad-package",
        )


def test_normalization_marks_empty_concerns_inapplicable() -> None:
    normalized = normalize_structured_state_section(_credits_section())
    inactive = {
        row["concern"]
        for row in normalized["specification"]["inapplicable_concerns"]
    }

    assert "transitions" in inactive
    assert "cleanup" in inactive
    assert "concurrency" in inactive
    assert "variables" not in inactive
