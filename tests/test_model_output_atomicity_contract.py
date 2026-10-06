from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import (
    assert_atomic_model_schema,
    assert_installed,
    install,
    is_atomic_model_schema,
    structured_output_token_ceiling,
    _model_transport_schema,
)


def test_large_closed_model_schema_is_allowed() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {
                    f"nested_{inner}": {"type": "string", "maxLength": 64}
                    for inner in range(4)
                },
                "additionalProperties": False,
            }
            for index in range(20)
        },
        "additionalProperties": False,
    }

    assert_atomic_model_schema(schema, surface="regression")
    assert is_atomic_model_schema(schema)


def test_small_atomic_schema_remains_allowed() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 64}},
        "required": ["value"],
        "additionalProperties": False,
    }

    assert_atomic_model_schema(schema, surface="regression")
    assert is_atomic_model_schema(schema)


def test_native_tool_decision_uses_the_same_atomicity_boundary() -> None:
    calls: list[str] = []

    class DummyRouter:
        def generate_text(self, role, messages, **kwargs):
            calls.append("text")
            return "ok"

        def generate_tool_decision(
            self,
            role,
            messages,
            *,
            tool_name,
            parameters,
            description="",
        ):
            calls.append("tool")
            return {"ok": True}

    module = SimpleNamespace(ModelRouter=DummyRouter)
    install(model_router_module=module)
    assert_installed(model_router_module=module)
    oversized = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {
                    f"nested_{inner}": {"type": "string", "maxLength": 64}
                    for inner in range(4)
                },
                "additionalProperties": False,
            }
            for index in range(20)
        },
        "additionalProperties": False,
    }

    result = DummyRouter().generate_tool_decision(
        "planner",
        ({"role": "user", "content": "fill it"},),
        tool_name="wide_planner_contract",
        parameters=oversized,
    )
    assert result == {"ok": True}
    assert calls == ["tool"]


def test_native_tool_decision_allows_bounded_closed_schema() -> None:
    calls: list[str] = []

    class DummyRouter:
        def generate_text(self, role, messages, **kwargs):
            return "ok"

        def generate_tool_decision(
            self,
            role,
            messages,
            *,
            tool_name,
            parameters,
            description="",
        ):
            calls.append(tool_name)
            return {"value": "ok"}

    module = SimpleNamespace(ModelRouter=DummyRouter)
    install(model_router_module=module)
    result = DummyRouter().generate_tool_decision(
        "planner",
        ({"role": "user", "content": "fill it"},),
        tool_name="bounded_planner_contract",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "maxLength": 64}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    assert result == {"value": "ok"}
    assert calls == ["bounded_planner_contract"]


@pytest.mark.parametrize(
    "kind",
    ["depth", "properties", "unbounded_string", "unbounded_array"],
)
def test_schema_shape_is_not_rejected_by_arbitrary_size_bounds(kind: str) -> None:
    if kind == "depth":
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "level4": {"type": "string", "maxLength": 32}
                                    },
                                    "additionalProperties": False,
                                }
                            },
                            "additionalProperties": False,
                        }
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }
    elif kind == "properties":
        schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string", "maxLength": 32}
                for i in range(10)
            },
            "additionalProperties": False,
        }
    elif kind == "unbounded_string":
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 32},
                }
            },
            "additionalProperties": False,
        }

    assert_atomic_model_schema(schema, surface=f"size-regression-{kind}")
    assert is_atomic_model_schema(schema)


def test_union_string_scalar_does_not_require_global_string_bound() -> None:
    bounded = {
        "type": "object",
        "properties": {
            "initializer": {
                "type": ["string", "number", "boolean", "null"],
                "maxLength": 64,
            }
        },
        "required": ["initializer"],
        "additionalProperties": False,
    }
    assert_atomic_model_schema(bounded, surface="union scalar")
    assert is_atomic_model_schema(bounded)

    unbounded = {
        "type": "object",
        "properties": {
            "initializer": {
                "type": ["string", "number", "boolean", "null"],
            }
        },
        "required": ["initializer"],
        "additionalProperties": False,
    }
    assert_atomic_model_schema(unbounded, surface="union scalar")
    assert is_atomic_model_schema(unbounded)


def test_open_object_template_still_rejected() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    with pytest.raises(ModelConfigurationError, match="MODEL_JSON_TEMPLATE_REQUIRED"):
        assert_atomic_model_schema(schema, surface="open template")
    assert not is_atomic_model_schema(schema)



@pytest.mark.parametrize(
    ("keyword", "fragment"),
    [
        (
            "uniqueItems",
            {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        ),
        (
            "contains",
            {
                "type": "array",
                "items": {"type": "string"},
                "contains": {"const": "x"},
            },
        ),
        (
            "multipleOf",
            {
                "type": "integer",
                "multipleOf": 2,
            },
        ),
    ],
)
def test_host_only_constraints_are_rejected_before_inference(
    keyword: str,
    fragment: dict,
) -> None:
    schema = {
        "type": "object",
        "properties": {"value": fragment},
        "required": ["value"],
        "additionalProperties": False,
    }

    with pytest.raises(
        ModelConfigurationError,
        match=f"MODEL_SCHEMA_HOST_ONLY_CONSTRAINT.*{keyword}",
    ):
        assert_atomic_model_schema(schema, surface="transport-parity-regression")


def test_model_transport_synthesizes_missing_resource_bounds() -> None:
    logical = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    }

    projected = _model_transport_schema(logical)

    assert projected["properties"]["value"]["maxLength"] == 256
    assert projected["properties"]["items"]["maxItems"] == 4
    assert projected["properties"]["items"]["items"]["maxLength"] == 256
    assert "maxLength" not in logical["properties"]["value"]
    assert "maxItems" not in logical["properties"]["items"]


def test_model_transport_preserves_explicit_domain_bounds() -> None:
    logical = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "maxLength": 4096},
            "items": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 1024},
            },
        },
        "additionalProperties": False,
    }

    projected = _model_transport_schema(logical)

    assert projected["properties"]["value"]["maxLength"] == 4096
    assert projected["properties"]["items"]["maxItems"] == 12
    assert projected["properties"]["items"]["items"]["maxLength"] == 1024


def test_structured_output_ceiling_is_schema_derived() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "maxLength": 32},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    ceiling = structured_output_token_ceiling(schema)

    assert 64 <= ceiling < 4096


def test_transport_projects_small_integer_range_to_finite_enum() -> None:
    logical = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
            }
        },
        "required": ["count"],
        "additionalProperties": False,
    }

    projected = _model_transport_schema(logical)

    assert projected["properties"]["count"]["enum"] == [0, 1, 2, 3, 4]
    assert structured_output_token_ceiling(projected) >= 64


def test_structured_output_ceiling_supports_oneof_and_local_ref() -> None:
    schema = {
        "$defs": {
            "short_text": {
                "type": "string",
                "maxLength": 12,
            }
        },
        "type": "object",
        "properties": {
            "choice": {
                "oneOf": [
                    {"$ref": "#/$defs/short_text"},
                    {"type": "boolean"},
                ]
            }
        },
        "required": ["choice"],
        "additionalProperties": False,
    }

    assert 64 <= structured_output_token_ceiling(schema) < 4096


def test_structured_output_ceiling_rejects_unbounded_numeric_lexical_surface() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    with pytest.raises(ValueError, match="unbounded lexical output"):
        structured_output_token_ceiling(schema)


def test_structured_output_ceiling_rejects_oversized_page_before_inference() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "maxLength": 1024},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    with pytest.raises(ValueError, match="exceeds the global atomic output bound"):
        structured_output_token_ceiling(schema)
