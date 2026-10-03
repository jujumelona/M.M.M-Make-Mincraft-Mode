from __future__ import annotations

import json

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_execution_schema import concern_contracts
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.implementation_graph_execution import _canonical_atomic_obligations
from minecraft_mod_ai.production_state_compiler import (
    _generate_concern_records,
    _normalize_expression,
    _parse_semantic_page,
    compile_production_state_section,
    normalize_structured_state_section,
)
from minecraft_mod_ai.structured_state_runtime import (
    render_state_model_concern,
    validate_state_expression,
)


class PlanRouter:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return self.response


class ProductionStateRouter:
    def __init__(self):
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        concern = payload["concern"]
        self.calls.append((role, concern, kwargs))
        records = {
            "variables": [
                {
                    "name": "credits",
                    "owner": "player",
                    "type": "integer",
                    "unit": "credits",
                    "default": "0",
                    "domain": "integer >= 0",
                }
            ],
            "transitions": [
                {
                    "from_state": "dock",
                    "trigger": "launch",
                    "guard": "shipStatus == ShipStatus.COMPLETE AND credits >= cost",
                    "mutation": "credits -= cost",
                    "to_state": "space",
                }
            ],
            "invariants": [],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
        }[concern]
        return json.dumps({"records": records, "complete": True})


def _plan_text() -> str:
    return (
        "# StarForge\n"
        "## behavior_contract\n"
        "- actors: player ship\n"
        "## state_model\n"
        "- variables: credits and ship state\n"
        "- transitions: launch only when Ship Status is Complete and enough credits exist\n"
        "## algorithm\n"
        "- steps: launch flow\n"
        "## integration\n"
        "- lifecycle: initialize\n"
        "## verification\n"
        "- tests: compile and launch\n"
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


def _structured_plan(section):
    return AuthoredPlan(
        "make a space mod",
        _plan_text(),
        structured_sections={"state_model": section},
    )


def test_free_markdown_state_is_rejected_before_production_model_decode():
    class PlannerRouter:
        def __init__(self):
            self.calls = []

        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, messages, kwargs))
            return _plan_text()

    planner_router = PlannerRouter()
    planner = CompleteGameDesignPlanner(planner_router)
    plan = planner.plan("make a space mod")

    assert plan.structured_sections == {}
    assert plan.text == _plan_text()
    assert len(planner_router.calls) == 1

    class ForbiddenStateRouter:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("production state compilation must not call the model")

    with pytest.raises(
        ValueError,
        match="PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED",
    ):
        compile_production_state_section(ForbiddenStateRouter(), plan)


def test_malformed_json_like_state_output_is_parsed_without_json_validation():
    raw = (
        '{"type": "object", "properties": {"records": ['
        '{"name": "credits";"owner": "player";"type": "double";'
        '"unit": "currency";"default": "0.0";"domain": "financial"}, '
        '{"name": "ship_state";"owner": "player";"type": "string";'
        '"unit": "status";"default": "\\\"Docked\\\"";"domain": "navigation"}'
        '], "complete": true}'
    )

    records, complete = _parse_semantic_page(
        raw,
        fields=("name", "owner", "type", "unit", "default", "domain"),
    )

    assert complete is True
    assert records == [
        {
            "name": "credits",
            "owner": "player",
            "type": "double",
            "unit": "currency",
            "default": "0.0",
            "domain": "financial",
        },
        {
            "name": "ship_state",
            "owner": "player",
            "type": "string",
            "unit": "status",
            "default": '"Docked"',
            "domain": "navigation",
        },
    ]


def test_production_state_compile_never_calls_router_when_structured_state_exists():
    class ForbiddenRouter:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("structured production state must be model-free")

    section = _state_section(
        variables=[{
            "name": "credits",
            "owner": "player",
            "type": "integer",
            "unit": "credits",
            "default": "0",
            "domain": "integer >= 0",
        }],
        invariants=[{
            "condition": "credits >= 0",
            "enforcement": "reject negative balances",
        }],
    )
    compiled = compile_production_state_section(
        ForbiddenRouter(),
        _structured_plan(section),
    )

    assert compiled["specification"]["invariants"] == [
        {
            "condition": "credits >= 0",
            "enforcement": "reject negative balances",
        }
    ]


def test_production_state_normalizes_structured_dsl_without_model():
    class ForbiddenRouter:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("structured production state must be model-free")

    plan = _structured_plan(_state_section(
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
                "name": "shipStatus",
                "owner": "player",
                "type": "string",
                "unit": "status",
                "default": "DOCKED",
                "domain": "status",
            },
        ],
        transitions=[{
            "from_state": "dock",
            "trigger": "launch",
            "guard": "shipStatus == ShipStatus.COMPLETE AND credits >= cost",
            "mutation": "credits -= cost",
            "to_state": "space",
        }],
    ))

    section = compile_production_state_section(ForbiddenRouter(), plan)

    transitions = section["specification"]["transitions"]
    assert transitions[0]["guard"] == 'shipStatus == "COMPLETE" && credits >= cost'
    assert transitions[0]["mutation"] == "credits -= cost"

    obligations = []
    for concern in ("variables", "transitions"):
        obligations.append(
            json.dumps(
                {
                    "instruction": json.dumps(
                        {"section": "state_model", "concern": concern}
                    ),
                    "structured_records": section["specification"][concern],
                }
            )
        )
    task = {"implementation_obligations": obligations}
    java = render_state_model_concern(task, "transitions", include_runtime=True)

    assert java is not None
    assert "ShipStatus.COMPLETE" not in java
    assert '$mmmRead("shipStatus", context)' in java
    assert '$mmmRead("credits", context)' in java
    assert '$mmmRead("cost", context)' in java


def test_multiword_state_expression_is_canonicalized_before_host_parser():
    variables = {
        "fuel_amount": {
            "name": "fuel_amount",
            "owner": "player",
            "type": "double",
            "unit": "fuel",
            "default": "100",
            "domain": "number",
        },
        "ship_state": {
            "name": "ship_state",
            "owner": "player",
            "type": "string",
            "unit": "status",
            "default": "Docked",
            "domain": "navigation status",
        },
    }

    normalized = _normalize_expression(
        "(fuel_amount >= Required Fuel) AND ship_state == Ready To Launch",
        aliases={},
        variables=variables,
        fallback="false",
    )

    assert "Required Fuel" not in normalized
    assert "Required_Fuel" in normalized
    assert '"Ready To Launch"' in normalized

    obligations = [
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "variables"}
            ),
            "structured_records": list(variables.values()),
        }),
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "invariants"}
            ),
            "structured_records": [
                {
                    "condition": normalized,
                    "enforcement": "block invalid launch",
                }
            ],
        }),
    ]
    java = render_state_model_concern(
        {"implementation_obligations": obligations},
        "invariants",
        include_runtime=True,
    )

    assert java is not None
    assert '$mmmRead("fuel_amount", context)' in java
    assert '$mmmRead("Required_Fuel", context)' in java
    assert '"Ready To Launch"' in java


def test_state_variable_spellings_resolve_to_declared_identity():
    variables = {
        "fuel_amount": {
            "name": "fuel_amount",
            "owner": "player",
            "type": "double",
            "unit": "fuel",
            "default": "100",
            "domain": "number",
        }
    }

    spaced = _normalize_expression(
        "Fuel Amount >= 10",
        aliases={},
        variables=variables,
        fallback="false",
    )
    camel = _normalize_expression(
        "fuelAmount >= 10",
        aliases={},
        variables=variables,
        fallback="false",
    )

    assert spaced == "fuel_amount >= 10"
    assert camel == "fuel_amount >= 10"


def test_unsupported_domain_function_fails_closed_before_java_lowering():
    variables = {
        "Ship_Status": {
            "name": "Ship_Status",
            "owner": "server",
            "type": "string",
            "unit": "status",
            "default": "WAITING",
            "domain": "status",
        }
    }

    normalized = _normalize_expression(
        'Ship_Position_DistFromCenter ( Planet_Earth ) <= 10.0 && '
        'Ship_Status == "WAITING"',
        aliases={},
        variables=variables,
        fallback="false",
    )

    assert normalized == "false"


def test_structured_state_section_rejects_unknown_function_semantics():
    import pytest
    from minecraft_mod_ai.structured_state_runtime import validate_structured_state_section

    section = {
        "specification": {
            "variables": [],
            "transitions": [],
            "invariants": [
                {
                    "condition": "Ship_Position_DistFromCenter(Planet_Earth) <= 10.0",
                    "enforcement": "hold position",
                }
            ],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
        }
    }

    with pytest.raises(ValueError, match="unsupported function 'Ship_Position_DistFromCenter'"):
        validate_structured_state_section(section)


def test_normalized_unknown_function_invariant_renders_without_crashing():
    raw = {
        "specification": {
            "variables": [],
            "transitions": [],
            "invariants": [
                {
                    "condition": "Ship_Position_DistFromCenter(Planet_Earth) <= 10.0",
                    "enforcement": "hold position",
                }
            ],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }

    normalized = normalize_structured_state_section(raw)
    assert normalized["specification"]["invariants"][0]["condition"] == "false"

    obligations = [
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "invariants"}
            ),
            "structured_records": normalized["specification"]["invariants"],
        })
    ]
    java = render_state_model_concern(
        {"implementation_obligations": obligations},
        "invariants",
        include_runtime=True,
    )

    assert java is not None
    assert "context -> ($mmmTruthy(Boolean.FALSE))" in java


def test_irreducible_state_condition_fails_closed_instead_of_crashing():
    variables = {
        "fuel_amount": {
            "name": "fuel_amount",
            "owner": "player",
            "type": "double",
            "unit": "fuel",
            "default": "100",
            "domain": "number",
        }
    }

    normalized = _normalize_expression(
        "fuel_amount >= )",
        aliases={},
        variables=variables,
        fallback="false",
    )

    assert normalized == "false"


def test_quoted_text_is_not_rewritten_as_state_dsl_syntax():
    variables = {
        "ship_state": {
            "name": "ship_state",
            "owner": "player",
            "type": "string",
            "unit": "status",
            "default": "Docked",
            "domain": "status",
        }
    }

    normalized = _normalize_expression(
        'ship_state == "READY   AND WAIT = SAFE"',
        aliases={},
        variables=variables,
        fallback="false",
    )

    assert normalized == 'ship_state == "READY   AND WAIT = SAFE"'


def test_semicolon_inside_state_string_literal_is_not_split_as_mutation():
    variables = {
        "ship_state": {
            "name": "ship_state",
            "owner": "player",
            "type": "string",
            "unit": "status",
            "default": "Docked",
            "domain": "status",
        }
    }

    from minecraft_mod_ai.production_state_compiler import _normalize_mutation

    mutation = _normalize_mutation(
        'ship_state = "READY;WAIT"',
        aliases={},
        variables=variables,
    )

    assert mutation == 'ship_state = "READY;WAIT"'

    obligations = [
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "variables"}
            ),
            "structured_records": list(variables.values()),
        }),
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "updates"}
            ),
            "structured_records": [
                {
                    "trigger": "tick",
                    "mutation": mutation,
                    "owner": "server",
                }
            ],
        }),
    ]
    java = render_state_model_concern(
        {"implementation_obligations": obligations},
        "updates",
        include_runtime=True,
    )

    assert java is not None
    assert '"READY;WAIT"' in java


def test_invalid_assignment_rhs_is_excluded_not_rewritten_to_self_assignment():
    from minecraft_mod_ai.production_state_compiler import _normalize_mutation

    variables = {
        "credits": {
            "name": "credits",
            "owner": "player",
            "type": "integer",
            "unit": "credits",
            "default": "0",
            "domain": "integer >= 0",
        }
    }

    mutation = _normalize_mutation(
        "credits = )",
        aliases={},
        variables=variables,
    )

    assert mutation is None


def test_host_expression_parser_accepts_trailing_whitespace_only():
    validate_state_expression("credits >= 0   ")


def test_cleanup_subsystem_action_is_dropped_by_host_normalizer_without_model():
    section = compile_production_state_section(
        None,
        _structured_plan(_state_section(
            variables=[{
                "name": "credits",
                "owner": "player",
                "type": "integer",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            }],
            cleanup=[{
                "event": "shutdown",
                "action": "SaveStateToFile",
                "retained_state": "player progress",
            }],
        )),
    )

    assert section["specification"]["cleanup"] == []
    assert {
        row["concern"]
        for row in section["specification"]["inapplicable_concerns"]
    } >= {"cleanup"}


def test_undeclared_mutation_target_is_not_promoted_to_state_variable():
    section = compile_production_state_section(
        None,
        _structured_plan(_state_section(
            variables=[{
                "name": "credits",
                "owner": "player",
                "type": "integer",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            }],
            updates=[{
                "trigger": "tick",
                "mutation": "ghost_counter += 1",
                "owner": "server",
            }],
        )),
    )

    assert [row["name"] for row in section["specification"]["variables"]] == ["credits"]
    assert section["specification"]["updates"] == []


def test_transition_without_state_assignment_uses_empty_program_not_magic_token():
    section = compile_production_state_section(
        None,
        _structured_plan(_state_section(
            variables=[{
                "name": "ship_state",
                "owner": "player",
                "type": "string",
                "unit": "status",
                "default": "Docked",
                "domain": "status",
            }],
            transitions=[{
                "from_state": "dock",
                "trigger": "launch",
                "guard": "ship_state == Ready",
                "mutation": "SendPacket",
                "to_state": "space",
            }],
        )),
    )

    transition = section["specification"]["transitions"][0]
    assert transition["mutation"] == ""
    assert "noop" not in json.dumps(section)


def test_production_state_sidecar_activates_state_leaf_without_text_anchor():
    section = {
        "specification": {
            "variables": [
                {
                    "name": "credits",
                    "owner": "player",
                    "type": "integer",
                    "unit": "credits",
                    "default": "0",
                    "domain": "integer >= 0",
                }
            ],
            "transitions": [],
            "invariants": [],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }

    obligations, drifted, active = _canonical_atomic_obligations(
        section="state_model",
        concerns=list(concern_contracts("state_model")),
        requirements={},
        raw_obligations=[],
        structured_sections={},
        production_state_section=section,
    )

    assert drifted == []
    assert [row["concern"] for row in active] == ["variables"]
    payload = json.loads(obligations[0])
    assert payload["source_requirements"] == {}
    assert payload["structured_records"] == section["specification"]["variables"]


def test_production_state_is_sidecar_and_does_not_mutate_authored_plan():
    raw = _state_section(
        variables=[{
            "name": "credits",
            "owner": "player",
            "type": "integer",
            "unit": "credits",
            "default": "0",
            "domain": "integer >= 0",
        }]
    )
    original = _structured_plan(raw)
    original_snapshot = original.to_dict()
    section = compile_production_state_section(None, original)

    modules, _manifest = _compile_new_authored_modules(
        original,
        mod_id="authored_test",
        package_name="ai.minecraft.generated.authored_test",
        target={},
        production_state_section=section,
    )

    request = modules[0].config["implementation_graph_request"]
    assert request["production_state_section"] == section
    assert request["structured_sections"] == original.structured_sections
    assert request["structured_sections_sha256"].startswith("sha256:")
    assert original.to_dict() == original_snapshot


def test_lark_state_expression_accepts_word_logic_aggregates_and_implication():
    for expression in (
        "current_ship_volume <= 50000 AND ship_module_slots_available > 0 OR trade_offer_validated",
        "player_resource_balance >= sum(required_costs_for_all_pending_module_constructions)",
        "interstellar_travel_sequence_active -> warp_gate_access_enabled = true",
        "NOT blocked OR (credits >= cost AND ready)",
    ):
        validate_state_expression(expression)


def test_structured_state_boundary_normalizes_latest_runtime_shapes():
    raw = {
        "specification": {
            "variables": [
                {
                    "name": "resource_balance",
                    "owner": "player",
                    "type": "double",
                    "unit": "credits",
                    "default": "0",
                    "domain": "number",
                },
                {
                    "name": "interstellar_travel_status",
                    "owner": "player",
                    "type": "boolean",
                    "unit": "status",
                    "default": "false",
                    "domain": "boolean",
                },
            ],
            "transitions": [],
            "invariants": [
                {
                    "condition": (
                        "current_ship_volume <= 50000 AND "
                        "ship_module_slots_available > 0 OR trade_offer_validated"
                    ),
                    "enforcement": "reject invalid state",
                },
                {
                    "condition": (
                        "player_resource_balance >= "
                        "sum(required_costs_for_all_pending_module_constructions)"
                    ),
                    "enforcement": "reject insufficient resources",
                },
                {
                    "condition": (
                        "interstellar_travel_sequence_active -> "
                        "warp_gate_access_enabled = true"
                    ),
                    "enforcement": "block early travel",
                },
            ],
            "initialization": [
                {
                    "owner": "game_engine",
                    "trigger": "start",
                    "initial_state": (
                        "interstellar_travel_status: false, "
                        "unknown_external_state: {}"
                    ),
                }
            ],
            "updates": [
                {
                    "trigger": "tick",
                    "mutation": (
                        "resource_balance += extracted_amount; "
                        "inventory.update()"
                    ),
                    "owner": "game_engine",
                }
            ],
            "cleanup": [],
            "concurrency": [],
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }

    normalized = normalize_structured_state_section(raw)
    spec = normalized["specification"]

    assert spec["invariants"][0]["condition"] == (
        "current_ship_volume <= 50000 && "
        "ship_module_slots_available > 0 || trade_offer_validated"
    )
    assert "sum" in spec["invariants"][1]["condition"]
    assert "->" in spec["invariants"][2]["condition"]
    assert spec["updates"] == [
        {
            "trigger": "tick",
            "mutation": "resource_balance += extracted_amount",
            "owner": "game_engine",
        }
    ]
    assert all(
        "inventory.update" not in row.get("mutation", "")
        for row in spec["updates"]
    )


def test_lark_state_expression_compiles_latest_invariants_without_model():
    obligations = [
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "variables"}
            ),
            "structured_records": [
                {
                    "name": "resource_balance",
                    "owner": "player",
                    "type": "double",
                    "unit": "credits",
                    "default": "0",
                    "domain": "number",
                }
            ],
        }),
        json.dumps({
            "instruction": json.dumps(
                {"section": "state_model", "concern": "invariants"}
            ),
            "structured_records": [
                {
                    "condition": (
                        "current_ship_volume <= 50000 AND "
                        "ship_module_slots_available > 0 OR trade_offer_validated"
                    ),
                    "enforcement": "reject",
                },
                {
                    "condition": (
                        "player_resource_balance >= "
                        "sum(required_costs_for_all_pending_module_constructions)"
                    ),
                    "enforcement": "reject",
                },
                {
                    "condition": (
                        "interstellar_travel_sequence_active -> "
                        "warp_gate_access_enabled = true"
                    ),
                    "enforcement": "reject",
                },
            ],
        }),
    ]
    java = render_state_model_concern(
        {"implementation_obligations": obligations},
        "invariants",
        include_runtime=True,
    )

    assert java is not None
    assert "$mmmFunction" in java
    assert "&&" in java
    assert "||" in java



def test_production_state_extractor_receives_only_selected_concern_block():
    class Router:
        def __init__(self):
            self.payload = None

        def generate_text(self, role, messages, **kwargs):
            self.payload = json.loads(messages[-1]["content"])
            return (
                "STATUS=DONE\nRECORD\n"
                "name=player_currency\n"
                "owner=Player\n"
                "type=long\n"
                "unit=crystals\n"
                "default=0\n"
                "domain=integer >= 0\nEND"
            )

    source = (
        "## state_model\n"
        "- variables: name owner type unit default domain: "
        "`player_currency` Player long crystals 0 integer\n"
        "- transitions: from_state trigger guard mutation to_state: "
        "DRAFT build_part true player_currency -= 1 IN_PROGRESS\n"
    )
    router = Router()
    rows = _generate_concern_records(
        router,
        source=source,
        concern="variables",
        declared_names=[],
    )

    assert rows[0]["name"] == "player_currency"
    assert router.payload is not None
    supplied = router.payload["approved_state_model"]
    assert router.payload["contains_authored_values"] is True
    assert "- variables:" in supplied
    assert "- transitions:" not in supplied


def test_free_markdown_nested_state_bullets_are_explicit_payload():
    from minecraft_mod_ai.production_state_compiler import (
        _concern_has_explicit_payload,
        _concern_source,
    )

    source = (
        "## state_model\n"
        "- **variables**:\n"
        "    - `star_balance`: `player` 소유, `long`, 기본값 `0L`, 범위 `0 ~ 2^63-1`\n"
        "- **transitions**:\n"
        "    - `from_state`: `ground_mode` -> `dock_building`\n"
        "    - `trigger`: `player_interaction`\n"
    )

    variables = _concern_source(source, "variables")
    transitions = _concern_source(source, "transitions")

    assert "`star_balance`" in variables
    assert "`from_state`" not in variables
    assert "`from_state`" in transitions
    assert _concern_has_explicit_payload(source, "variables") is True
    assert _concern_has_explicit_payload(source, "transitions") is True


def test_free_markdown_nested_state_values_is_rejected_at_production_boundary():
    class ForbiddenRouter:
        def __init__(self):
            self.calls = []

        def generate_text(self, *_args, **_kwargs):
            self.calls.append(True)
            raise AssertionError("production must not reinterpret free Markdown state")

    plan = AuthoredPlan(
        "space mod",
        "## state_model\n- variables: free-form authored state\n",
    )
    router = ForbiddenRouter()

    with pytest.raises(
        ValueError,
        match="PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED",
    ):
        compile_production_state_section(router, plan)
    assert router.calls == []


def test_logged_free_markdown_state_is_rejected_without_model_decode():
    class ForbiddenRouter:
        def __init__(self):
            self.calls = []

        def generate_text(self, *_args, **_kwargs):
            self.calls.append(True)
            raise AssertionError("production must not reinterpret free Markdown state")

    plan = AuthoredPlan(
        "space mod",
        "## state_model\n- variables: free-form authored state\n",
    )
    router = ForbiddenRouter()

    with pytest.raises(
        ValueError,
        match="PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED",
    ):
        compile_production_state_section(router, plan)
    assert router.calls == []


def test_production_state_extractor_rejects_false_empty_for_explicit_authored_values():
    class EmptyRouter:
        def generate_text(self, role, messages, **kwargs):
            return "STATUS=EMPTY"

    source = (
        "## state_model\n"
        "- variables: name owner type unit default domain: "
        "`player_ship` Player enum status DRAFT DRAFT|IN_PROGRESS|READY|DESTROYED\n"
    )

    try:
        _generate_concern_records(
            EmptyRouter(),
            source=source,
            concern="variables",
            declared_names=[],
        )
    except ValueError as exc:
        assert "PRODUCTION_STATE_LOWERING_FALSE_EMPTY" in str(exc)
    else:
        raise AssertionError("explicit authored state must never be accepted as EMPTY")


def test_legacy_bold_state_concern_labels_are_detected_before_compat_lowering():
    from minecraft_mod_ai.production_state_compiler import (
        _concern_has_explicit_payload,
        _concern_source,
    )

    source = (
        "## state_model\n"
        "- **variables**: name owner type unit default domain\n"
        "  - record_1: name=credits; owner=player; type=integer; "
        "unit=credits; default=0; domain=integer >= 0\n"
        "- **transitions**: from_state trigger guard mutation to_state\n"
    )

    block = _concern_source(source, "variables")

    assert block.startswith("- **variables**:")
    assert "name=credits" in block
    assert _concern_has_explicit_payload(source, "variables") is True
