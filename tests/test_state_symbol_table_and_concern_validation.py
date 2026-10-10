from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.authored_structured_design import author_structured_sections
from minecraft_mod_ai.structured_output import StructuredOutputValidationError
from minecraft_mod_ai.structured_state_runtime import (
    StateSymbolTable,
    validate_state_concern,
)


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
            "guard": {
                "kind": "compare", "op": ">=",
                "left": {"kind": "state_ref", "name": "player_currency"},
                "right": {"kind": "state_ref", "name": "trade_cost"},
            },
            "mutation": [{
                "target": "player_currency", "operator": "-=",
                "value": {"kind": "state_ref", "name": "trade_cost"},
            }],
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
                "guard": {"kind": "literal", "value": True},
                "mutation": [{
                    "target": "current_phase", "operator": "=",
                    "value": {"kind": "number", "value": "1"},
                }],
                "to_state": "dock",
            }],
            symbols=symbols,
        )


def test_validate_state_concern_other_mutations():
    symbols = StateSymbolTable([{"name": "score"}])

    # Valid initialization
    validate_state_concern("initialization", [{
        "initial_state": [{
            "target": "score", "operator": "=",
            "value": {"kind": "number", "value": "0"},
        }]
    }], symbols=symbols)

    # Undeclared in updates fails immediately
    with pytest.raises(ValueError, match="undeclared state variable 'ghost_var'"):
        validate_state_concern("updates", [{
            "mutation": [{
                "target": "ghost_var", "operator": "+=",
                "value": {"kind": "number", "value": "1"},
            }]
        }], symbols=symbols)

    # Undeclared in cleanup fails immediately
    with pytest.raises(ValueError, match="undeclared state variable 'ghost_var'"):
        validate_state_concern("cleanup", [{
            "action": [{
                "target": "ghost_var", "operator": "=",
                "value": {"kind": "number", "value": "0"},
            }]
        }], symbols=symbols)


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
    target_values = [
        branch["properties"]["target"]["const"]
        for branch in schema_mut["items"]["oneOf"]
    ]
    assert target_values == ["player_currency", "ship_phase"]

    schema_expr = state_expr_schema(symbols)
    state_ref_branch = next(
        b for b in schema_expr["oneOf"]
        if b["properties"]["kind"].get("const") == "state_ref"
    )
    assert state_ref_branch["properties"]["name"]["enum"] == ["player_currency", "ship_phase"]


def test_author_structured_sections_passes_symbols_and_fails_undeclared_early():
    seen_prompts = []
    seen_schemas = []

    variable_values = {
        "name": "player_currency",
        "owner": "player",
        "type": "int",
        "unit": "credits",
        "default": "0",
        "domain": "int >= 0",
    }
    transition_values = {
        "from_state": "dock",
        "trigger": "trade",
        "guard": "player_currency >= 0",
        "mutation": "current_phase = 1",
        "to_state": "dock",
    }

    def schema_value(schema):
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        raw_type = schema.get("type")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        if "string" in types:
            return "x"
        if "integer" in types:
            return 1
        if "number" in types:
            return 1.0
        if "boolean" in types:
            return True
        if "array" in types:
            minimum = int(schema.get("minItems", 0) or 0)
            return [schema_value(schema.get("items", {})) for _ in range(minimum)]
        if "null" in types:
            return None
        raise AssertionError(f"unsupported fixture schema: {schema!r}")

    class MockRouter:
        def generate_text(self, role, messages, **kwargs):
            assert role == "planner"
            content = "\n".join(message["content"] for message in messages)
            seen_prompts.append(content)
            schema = kwargs.get("response_schema", {})
            seen_schemas.append(schema)
            properties = schema.get("properties", {})

            if "Author state mutation for" in content or any(f in properties for f in ("mutation", "initial_state", "action")):
                field = next(f for f in ("mutation", "initial_state", "action") if f in properties)
                return json.dumps({field: [{"target": "current_phase", "operator": "=", "value": {"kind": "literal", "value": "1"}}]})
            if "Author state expression for" in content or any(f in properties for f in ("guard", "condition")):
                field = next(f for f in ("guard", "condition") if f in properties)
                return json.dumps({field: {"kind": "literal", "value": "true"}})

            if "record_count" in properties:
                section = content.split("Section: ", 1)[1].splitlines()[0]
                concern = content.split("Concern: ", 1)[1].splitlines()[0]
                if section == "state_model":
                    count = 1 if concern in {"variables", "transitions"} else 0
                else:
                    count = 1
                return json.dumps({"record_count": count})

            required = list(schema.get("required", ()))
            assert len(required) == 1
            concern = required[0]
            concern_schema = properties[concern]
            min_items = int(concern_schema.get("minItems", 0) or 0)
            max_items = int(concern_schema.get("maxItems", min_items) or min_items)
            if min_items == max_items:
                count = min_items
            else:
                section = content.split("Section: ", 1)[1].splitlines()[0]
                if section == "state_model":
                    count = 1 if concern in {"variables", "transitions"} else 0
                else:
                    count = 1
            item_schema = concern_schema["items"]
            fields = list(item_schema.get("required", ()))
            assert 1 <= len(fields) <= 3

            if "Section: state_model" in content and concern == "variables":
                record = {f: variable_values[f] for f in fields}
            elif "Section: state_model" in content and concern == "transitions":
                record = {f: transition_values[f] for f in fields}
            else:
                record = {f: schema_value(item_schema["properties"][f]) for f in fields}

            return json.dumps({concern: [record for _ in range(count)]})

    router = MockRouter()

    with pytest.raises(
        StructuredOutputValidationError,
        match="current_phase.*not one of",
    ):
        author_structured_sections(router, "create a mod with trading")

    # The target decision itself rejects undeclared names before another field
    # or section can consume a malformed state assignment.
    targets = [
        schema["properties"][f]["items"]["properties"]["target"]
        for schema in seen_schemas
        for f in ("mutation", "initial_state", "action")
        if f in schema.get("properties", {})
        and "items" in schema["properties"][f]
        and "target" in schema["properties"][f]["items"].get("properties", {})
    ]
    assert targets and all(s["enum"] == ["player_currency"] for s in targets)

    # The canonical variable symbol table is still supplied to every later
    # transition field page before host-side validation rejects current_phase.
    transition_prompts = [
        prompt for prompt in seen_prompts
        if "Active Concerns: transitions" in prompt
    ]
    assert transition_prompts
    assert all("Canonical state symbols:" in prompt for prompt in transition_prompts)
    assert all("- player_currency" in prompt for prompt in transition_prompts)



def test_mutation_schema_resolves_symbols_before_building_value_branches():
    from minecraft_mod_ai.structured_state_runtime import (
        StateSymbolTable,
        mutations_schema,
    )

    symbols = StateSymbolTable(
        [
            {
                "name": "hull",
                "owner": "ship",
                "type": "double",
                "unit": "points",
                "default": "100",
                "domain": "0..100",
            },
            {
                "name": "status",
                "owner": "ship",
                "type": "string",
                "unit": "state",
                "default": "docked",
                "domain": "text",
            },
        ]
    )

    schema = mutations_schema(symbols)
    branches = schema["items"]["oneOf"]

    hull = next(
        branch
        for branch in branches
        if branch["properties"]["target"].get("const") == "hull"
    )
    value_branches = hull["properties"]["value"]["oneOf"]
    state_ref = next(
        branch
        for branch in value_branches
        if branch.get("properties", {}).get("kind", {}).get("const") == "state_ref"
    )

    assert state_ref["properties"]["name"]["enum"] == ["hull"]
