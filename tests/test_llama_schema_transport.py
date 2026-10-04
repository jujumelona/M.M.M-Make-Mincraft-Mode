from types import SimpleNamespace

from minecraft_mod_ai.llama_schema_transport import project_llama_transport_schema
from minecraft_mod_ai.llama_structured_decode_policy import (
    _apply_llama_json_schema,
    _is_qwen35,
)


def _request(schema):
    return SimpleNamespace(response_format="json", response_schema=schema)


def test_qwen35_receives_native_structural_json_sampler_constraints():
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        )
    )
    payload = {
        "response_format": {"type": "json_object"},
        "json_schema": {"type": "object"},
        "grammar": "legacy",
    }
    schema = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": ["note"],
    }

    assert _is_qwen35(adapter)
    _apply_llama_json_schema(payload, _request(schema), adapter=adapter)

    assert payload["response_format"] == {"type": "json_object"}
    assert payload["json_schema"] == project_llama_transport_schema(schema)
    assert "grammar" not in payload


def test_non_qwen_receives_only_structural_transport_schema():
    adapter = SimpleNamespace(
        config=SimpleNamespace(model_id="other/llama-model", extra={})
    )
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "minLength": 3, "pattern": "^[a-z]+$"},
            "items": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 2,
            },
        },
        "required": ["status", "items"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ready"}}},
                "then": {"properties": {"items": {"minItems": 5}}},
            }
        ],
    }
    payload = {}

    _apply_llama_json_schema(payload, _request(schema), adapter=adapter)

    assert payload["response_format"] == {"type": "json_object"}
    assert payload["json_schema"] == {
        "type": "object",
        "properties": {
            "status": {"type": "string", "minLength": 3, "pattern": "^[a-z]+$"},
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
        },
        "required": ["status", "items"],
        "additionalProperties": False,
    }


def test_explicit_object_structure_wins_over_allof_branch():
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "payload": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
        "required": ["kind", "payload"],
        "allOf": [
            {
                "if": {"properties": {"kind": {"const": "x"}}},
                "then": {"properties": {"payload": {"required": ["name"]}}},
            }
        ],
    }

    projected = project_llama_transport_schema(schema)

    assert projected["type"] == "object"
    assert set(projected["properties"]) == {"kind", "payload"}
    assert projected["required"] == ["kind", "payload"]
    assert "allOf" not in projected
    assert "if" not in str(projected)
    assert "then" not in str(projected)
    assert "const" not in str(projected)


def test_host_only_validation_keywords_are_removed_recursively():
    schema = {
        "type": "array",
        "minItems": 3,
        "items": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
            },
            "required": ["value"],
        },
    }

    assert project_llama_transport_schema(schema) == {
        "type": "array",
        "minItems": 3,
        "items": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
    }


def test_state_expr_schema_preserves_discriminated_oneof_and_variant_required_fields():
    from minecraft_mod_ai.structured_state_runtime import state_expr_schema

    schema = state_expr_schema(["ship_hull", "shield_level"])
    projected = project_llama_transport_schema(schema)

    assert "oneOf" in projected
    branches = projected["oneOf"]

    # state_ref variant
    state_ref = next(
        b for b in branches
        if b.get("properties", {}).get("kind", {}).get("const") == "state_ref"
    )
    assert state_ref["type"] == "object"
    assert state_ref["required"] == ["kind", "name"]
    assert state_ref["properties"]["kind"]["const"] == "state_ref"
    assert state_ref["properties"]["name"]["enum"] == ["shield_level", "ship_hull"]

    # literal variant
    literal = next(
        b for b in branches
        if b.get("properties", {}).get("kind", {}).get("const") == "literal"
    )
    assert literal["type"] == "object"
    assert literal["required"] == ["kind", "value"]
    assert literal["properties"]["kind"]["const"] == "literal"

    # compare variant
    compare = next(
        b for b in branches
        if b.get("properties", {}).get("kind", {}).get("const") == "compare"
    )
    assert compare["type"] == "object"
    assert compare["required"] == ["kind", "op", "left", "right"]
    assert compare["properties"]["kind"]["const"] == "compare"

    # number variant
    number = next(
        b for b in branches
        if b.get("properties", {}).get("kind", {}).get("const") == "number"
    )
    assert number["type"] == "object"
    assert number["required"] == ["kind", "value"]
    assert number["properties"]["kind"]["const"] == "number"

    # call variant
    call = next(
        b for b in branches
        if b.get("properties", {}).get("kind", {}).get("const") == "call"
    )
    assert call["type"] == "object"
    assert call["required"] == ["kind", "name", "args"]
    assert call["properties"]["kind"]["const"] == "call"
    assert call["properties"]["name"]["enum"] == [
        "abs", "count", "len", "max", "min", "size", "sum"
    ]


def test_pattern_plus_maxlength_synthesizes_bounded_pattern():
    """When both pattern and maxLength coexist, unbounded quantifiers in the
    pattern are replaced with bounded versions so llama.cpp enforces both
    grammar and length in a single grammar pass.
    """
    schema = {
        "type": "string",
        "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
        "maxLength": 128,
        "minLength": 1,
    }

    projected = project_llama_transport_schema(schema)

    # Both pattern (now bounded) and maxLength are preserved.
    assert projected["pattern"] == "^[a-z0-9_.-]{1,128}:[a-z0-9_./-]{1,128}$"
    assert projected["maxLength"] == 128
    assert projected["minLength"] == 1


def test_pattern_only_without_maxlength_is_preserved():
    """If pattern is the only string constraint, pass it through unchanged."""
    schema = {
        "type": "string",
        "pattern": r"^[a-z]+$",
    }

    projected = project_llama_transport_schema(schema)

    assert projected == {"type": "string", "pattern": "^[a-z]+$"}
    assert "maxLength" not in projected


def test_integer_transport_schema_preserves_bounded_pattern():
    """_model_transport_schema converts integers to string+pattern+maxLength.

    The integer pattern already uses bounded quantifiers ({0,18}), so
    _bound_pattern_quantifiers leaves it unchanged.  Both pattern and
    maxLength survive into the effective schema.
    """
    from minecraft_mod_ai.model_output_atomicity_contract import (
        effective_model_transport_schema,
    )

    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
        },
        "required": ["count"],
        "additionalProperties": False,
    }

    effective = effective_model_transport_schema(schema)

    count_schema = effective["properties"]["count"]
    assert count_schema["type"] == "string"
    assert count_schema["maxLength"] == 20
    # Integer pattern has no unbounded quantifiers — preserved unchanged.
    assert count_schema["pattern"] == r"^-?(?:0|[1-9][0-9]{0,18})$"


def test_identifier_pattern_is_bounded_by_maxlength():
    """Context ref names use pattern + maxLength — both must be enforced."""
    schema = {
        "type": "string",
        "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]*$",
        "maxLength": 24,
    }

    projected = project_llama_transport_schema(schema)

    assert projected["pattern"] == "^[A-Za-z_$][A-Za-z0-9_$.]{0,24}$"
    assert projected["maxLength"] == 24

