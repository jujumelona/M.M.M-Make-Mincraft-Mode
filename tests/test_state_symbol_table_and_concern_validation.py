from __future__ import annotations

import json
import pytest

from minecraft_mod_ai.structured_state_runtime import (
    StateSymbolTable,
    validate_state_concern,
)
from minecraft_mod_ai.authored_structured_design import author_structured_sections


def test_state_symbol_table_basics():
    table = StateSymbolTable([
        {"name": "player_currency", "type": "int"},
        {"name": "ship_blueprint", "type": "string"},
    ])

    assert table.contains("player_currency")
    assert "ship_blueprint" in table
    assert "current_phase" not in table
    assert len(table) == 2
    assert list(table) == ["player_currency", "ship_blueprint"]

    prompt = table.prompt_text()
    assert "Canonical state symbols:" in prompt
    assert "variables:" in prompt
    assert "- player_currency" in prompt
    assert "- ship_blueprint" in prompt


def test_validate_state_concern_variables():
    # Valid variables pass
    validate_state_concern("variables", [
        {"name": "player_currency"},
        {"name": "ship_blueprint"},
    ])

    # Invalid identifier fails
    with pytest.raises(ValueError, match="STRUCTURED_STATE_VARIABLE_NAME"):
        validate_state_concern("variables", [{"name": "not an id!"}])

    # Duplicate identifier fails
    with pytest.raises(ValueError, match="STRUCTURED_STATE_VARIABLE_DUPLICATE"):
        validate_state_concern("variables", [
            {"name": "player_currency"},
            {"name": "player_currency"},
        ])


def test_validate_state_concern_transitions_with_symbols():
    symbols = StateSymbolTable([{"name": "player_currency"}, {"name": "trade_cost"}])

    # Valid guard and mutation pass
    validate_state_concern(
        "transitions",
        [{
            "from_state": "dock",
            "trigger": "buy",
            "guard": "player_currency >= trade_cost",
            "mutation": "player_currency -= trade_cost",
            "to_state": "dock",
        }],
        symbols=symbols,
    )

    # Undeclared mutation target current_phase fails immediately!
    with pytest.raises(
        ValueError,
        match="STRUCTURED_STATE_MUTATION: undeclared state variable 'current_phase'",
    ):
        validate_state_concern(
            "transitions",
            [{
                "from_state": "dock",
                "trigger": "trade",
                "guard": "true",
                "mutation": "current_phase = 1",
                "to_state": "dock",
            }],
            symbols=symbols,
        )


def test_validate_state_concern_other_mutations():
    symbols = StateSymbolTable([{"name": "score"}])

    # Valid initialization
    validate_state_concern("initialization", [{"initial_state": "score = 0"}], symbols=symbols)

    # Undeclared in updates fails immediately
    with pytest.raises(ValueError, match="undeclared state variable 'ghost_var'"):
        validate_state_concern("updates", [{"mutation": "ghost_var += 1"}], symbols=symbols)

    # Undeclared in cleanup fails immediately
    with pytest.raises(ValueError, match="undeclared state variable 'ghost_var'"):
        validate_state_concern("cleanup", [{"action": "ghost_var = 0"}], symbols=symbols)


def test_structured_ir_validation_and_compilation():
    from minecraft_mod_ai.structured_state_runtime import (
        compile_mutation_ir,
        compile_state_expr_ir,
        mutations_schema,
        state_expr_schema,
        validate_mutation_ir,
        validate_state_expr_ir,
    )

    symbols = StateSymbolTable([{"name": "player_currency"}, {"name": "ship_phase"}])

    # 1. Structured guard IR
    guard_ir = {
        "kind": "and",
        "terms": [
            {
                "kind": "compare",
                "op": ">=",
                "left": {"kind": "state_ref", "name": "player_currency"},
                "right": {"kind": "context_ref", "name": "cost"},
            }
        ],
    }
    validate_state_expr_ir(guard_ir, symbols=symbols)
    java_guard = compile_state_expr_ir(guard_ir, declared=set(symbols))
    assert "$mmmCompare($mmmRead(\"player_currency\", context), $mmmRead(\"cost\", context)) >= 0" in java_guard

    # 2. Structured guard with undeclared variable fails
    invalid_guard = {
        "kind": "compare",
        "op": "==",
        "left": {"kind": "state_ref", "name": "undeclared_var"},
        "right": {"kind": "literal", "value": 1},
    }
    with pytest.raises(ValueError, match="undeclared state variable 'undeclared_var'"):
        validate_state_expr_ir(invalid_guard, symbols=symbols)

    # 3. Structured mutation IR
    mutations = [
        {
            "target": "ship_phase",
            "operator": "=",
            "value": {"kind": "literal", "value": 1},
        },
        {
            "target": "player_currency",
            "operator": "-=",
            "value": {"kind": "context_ref", "name": "cost"},
        },
    ]
    validate_mutation_ir(mutations, symbols=symbols)
    java_mut = compile_mutation_ir(mutations, declared=set(symbols))
    assert 'setState("ship_phase", Double.valueOf(1));' in java_mut
    assert 'setState("player_currency", $mmmArithmetic("-", $mmmRead("player_currency", context), $mmmRead("cost", context)));' in java_mut

    # 4. Structured mutation with undeclared target fails
    invalid_mut = [
        {
            "target": "unknown_phase",
            "operator": "=",
            "value": {"kind": "literal", "value": 1},
        }
    ]
    with pytest.raises(ValueError, match="undeclared state variable 'unknown_phase'"):
        validate_mutation_ir(invalid_mut, symbols=symbols)

    # 5. Schema generates enum from symbol table
    schema_mut = mutations_schema(symbols)
    target_enum = schema_mut["items"]["properties"]["target"]["enum"]
    assert target_enum == ["player_currency", "ship_phase"]

    schema_expr = state_expr_schema(symbols)
    op_name_enum = schema_expr["properties"]["left"]["properties"]["name"]["enum"]
    assert op_name_enum == ["player_currency", "ship_phase"]


def test_author_structured_sections_passes_symbols_and_fails_undeclared_early():
    seen_prompts = []
    seen_schemas = []

    class MockRouter:
        def generate_text(self, role, messages, **kwargs):
            content = messages[-1]["content"]
            seen_prompts.append(content)
            schema = kwargs.get("response_schema", {})
            seen_schemas.append(schema)

            # Check if this is state_model
            if "Section: state_model" in content:
                props = schema.get("properties", {})
                if "variables" in props:
                    return '{"variables": [{"name": "player_currency", "owner": "player", "type": "int", "unit": "credits", "default": "0", "domain": "int >= 0"}]}'
                if "transitions" in props:
                    # Attempt to use undeclared variable current_phase
                    return '{"transitions": [{"from_state": "dock", "trigger": "trade", "guard": {"kind": "literal", "value": true}, "mutations": [{"target": "current_phase", "operator": "=", "value": {"kind": "literal", "value": 1}}], "to_state": "dock"}]}'
                return '{"inapplicable_concerns": []}'

            # Non-state section response
            from worksheet_fixtures import row
            section_name = content.split("Section: ", 1)[1].splitlines()[0]
            full = row(section_name)
            payload = {}
            for prop in schema.get("properties", {}):
                if prop in full["specification"]:
                    payload[prop] = full["specification"][prop]
                elif prop in full:
                    payload[prop] = full[prop]
            return json.dumps(payload)

    router = MockRouter()

    with pytest.raises(
        (ValueError, RuntimeError),
        match=r"(STRUCTURED_STATE_MUTATION: undeclared state variable 'current_phase'|'current_phase' is not one of)",
    ):
        author_structured_sections(router, "create a mod with trading")

    # Verify that transition schema received the enum constraint from StateSymbolTable!
    transition_schemas = [
        s for s in seen_schemas
        if "transitions" in s.get("properties", {})
    ]
    assert len(transition_schemas) >= 1
    t_schema = transition_schemas[0]["properties"]["transitions"]["items"]
    target_enum = t_schema["properties"]["mutations"]["items"]["properties"]["target"]["enum"]
    assert target_enum == ["player_currency"]

    # Verify that transition prompt received the canonical state symbols from variables!
    transition_prompts = [p for p in seen_prompts if "Active Concerns: transitions" in p]
    assert len(transition_prompts) >= 1
    assert "Canonical state symbols:" in transition_prompts[0]
    assert "- player_currency" in transition_prompts[0]
