from types import SimpleNamespace

from minecraft_mod_ai.llama_schema_transport import project_llama_transport_schema
from minecraft_mod_ai.llama_structured_decode_policy import (
    _apply_llama_json_schema,
    _is_qwen35,
)


def _request(schema):
    return SimpleNamespace(response_format="json", response_schema=schema)


def test_qwen35_never_receives_native_json_sampler_constraints():
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

    assert "response_format" not in payload
    assert "json_schema" not in payload
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
            "status": {"type": "string"},
            "items": {"type": "array", "items": {"type": "integer"}},
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
        "items": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
    }
