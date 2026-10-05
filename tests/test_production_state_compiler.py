from __future__ import annotations

import copy

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.production_state_compiler import (
    compile_production_state_section,
    normalize_structured_state_section,
    render_production_state_java,
)


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


def _credits_section() -> dict:
    return _state_section(
        variables=[
            {
                "name": "credits",
                "owner": "player",
                "type": "long",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            }
        ],
        invariants=[
            {
                "condition": {
                    "kind": "compare",
                    "op": ">=",
                    "left": {"kind": "state_ref", "name": "credits"},
                    "right": {"kind": "number", "value": "0"},
                },
                "enforcement": "reject negative balances",
            }
        ],
        initialization=[
            {
                "owner": "player",
                "trigger": "player_join",
                "initial_state": [
                    {
                        "target": "credits",
                        "operator": "=",
                        "value": {"kind": "number", "value": "0"},
                    }
                ],
            }
        ],
        updates=[
            {
                "trigger": "reward",
                "mutation": [
                    {
                        "target": "credits",
                        "operator": "+=",
                        "value": {"kind": "number", "value": "1"},
                    }
                ],
                "owner": "server",
            }
        ],
    )


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


def test_missing_state_section_returns_empty_sidecar() -> None:
    assert compile_production_state_section(_plan()) == {}


def test_compile_uses_canonical_structured_authority_without_mutation() -> None:
    raw = _credits_section()
    plan = _plan(raw)
    before = copy.deepcopy(plan.structured_sections)

    compiled = compile_production_state_section(plan)

    assert compiled["specification"]["variables"][0]["name"] == "credits"
    assert compiled["specification"]["invariants"][0]["condition"]["kind"] == "compare"
    assert plan.structured_sections == before


def test_direct_specification_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="PRODUCTION_STATE_SPECIFICATION_REQUIRED"):
        normalize_structured_state_section(_credits_section()["specification"])


def test_invalid_state_identifier_is_not_normalized() -> None:
    section = _credits_section()
    section["specification"]["variables"][0]["name"] = "Ship Status"

    with pytest.raises(ValueError, match="STRUCTURED_STATE_VARIABLE_NAME"):
        normalize_structured_state_section(section)


def test_string_expression_is_rejected_instead_of_reparsed() -> None:
    section = _credits_section()
    section["specification"]["invariants"][0]["condition"] = "credits >= 0"

    with pytest.raises(ValueError, match="STRUCTURED_STATE_EXPRESSION"):
        normalize_structured_state_section(section)


def test_external_subsystem_action_is_rejected_as_state_mutation() -> None:
    section = _credits_section()
    section["specification"]["updates"][0]["mutation"] = "SendPacket"

    with pytest.raises(ValueError, match="STRUCTURED_STATE_MUTATION"):
        normalize_structured_state_section(section)


def test_unknown_typed_ir_function_fails_closed() -> None:
    section = _credits_section()
    section["specification"]["invariants"][0]["condition"] = {
        "kind": "call",
        "name": "mystery",
        "args": [{"kind": "state_ref", "name": "credits"}],
    }

    with pytest.raises(ValueError, match="unsupported function"):
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
    assert 'setState("credits", Double.valueOf("0"));' in source
    assert '$mmmArithmetic("+", $mmmRead("credits", context), Double.valueOf("1"))' in source


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
