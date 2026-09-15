from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import (
    assert_atomic_model_schema,
    assert_installed,
    install,
    is_atomic_model_schema,
)


def test_large_closed_model_schema_is_rejected() -> None:
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

    with pytest.raises(ModelConfigurationError):
        assert_atomic_model_schema(schema, surface="regression")
    assert not is_atomic_model_schema(schema)


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

    with pytest.raises(ModelConfigurationError):
        DummyRouter().generate_tool_decision(
            "planner",
            ({"role": "user", "content": "fill it"},),
            tool_name="oversized_planner_contract",
            parameters=oversized,
        )
    assert calls == []


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
def test_schema_violating_bounds_is_rejected(kind: str) -> None:
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

    with pytest.raises(ModelConfigurationError):
        assert_atomic_model_schema(schema, surface=f"bound-violation-{kind}")
    assert not is_atomic_model_schema(schema)


def test_open_object_template_still_rejected() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    with pytest.raises(ModelConfigurationError, match="MODEL_JSON_TEMPLATE_REQUIRED"):
        assert_atomic_model_schema(schema, surface="open template")
    assert not is_atomic_model_schema(schema)
