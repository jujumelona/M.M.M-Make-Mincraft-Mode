from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
    PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING,
    PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING,
)
from minecraft_mod_ai.authored_structured_design import (
    _PlannerPageRequest,
    _generate_authored_chunk,
    _generate_concern_record_count,
)
from minecraft_mod_ai.fixed_template_generation import (
    _generate_native_template_arguments,
    generate_fixed_template_value,
)
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.worksheet_atomic_chunker import (
    pack_section_concerns,
    planner_page_output_token_ceiling,
)


_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "minLength": 1, "maxLength": 32},
    },
    "required": ["value"],
    "additionalProperties": False,
}


class _Registry:
    @staticmethod
    def role(_profile: str, _role: str):
        return SimpleNamespace(adapter="llama_cpp")


class _PlannerRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.text_calls: list[tuple[tuple[dict, ...], dict]] = []
        self.tool_calls = 0

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        copied = tuple(dict(message) for message in messages)
        self.text_calls.append((copied, dict(kwargs)))
        return json.dumps({"value": "ok"})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner fixed templates must never use native tools")


def test_planner_fixed_template_uses_schema_json_not_native_tool() -> None:
    router = _PlannerRouter()

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "author one bounded page"},),
        response_schema=_SCHEMA,
        enable_tools=False,
    )

    assert value == {"value": "ok"}
    assert router.tool_calls == 0
    assert len(router.text_calls) == 1
    _messages, kwargs = router.text_calls[0]
    assert kwargs["response_format"] == "json"
    assert kwargs["response_schema"] == _SCHEMA
    assert kwargs["enable_tools"] is False
    assert kwargs["force_non_thinking"] is True
    assert kwargs["output_token_ceiling"] == ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING


def test_planner_fixed_template_honors_host_page_ceiling() -> None:
    router = _PlannerRouter()

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "author one bounded page"},),
        response_schema=_SCHEMA,
        enable_tools=False,
        output_token_ceiling=777,
    )

    assert value == {"value": "ok"}
    assert router.tool_calls == 0
    assert router.text_calls[0][1]["output_token_ceiling"] == 777


def test_planner_router_keeps_schema_out_of_model_messages() -> None:
    router = object.__new__(ModelRouter)
    schema = {
        "type": "object",
        "properties": {
            "private_transport_field": {
                "type": "string",
                "minLength": 1,
                "maxLength": 31,
            }
        },
        "required": ["private_transport_field"],
        "additionalProperties": False,
    }

    _stage, _runtime, tools, request = router._prepare_generation_request_impl(
        "planner",
        ({"role": "user", "content": "fill the bounded leaf"},),
        config=SimpleNamespace(adapter="llama_cpp"),
        response_format="json",
        response_schema=schema,
        enable_tools=False,
        output_token_ceiling=333,
        force_non_thinking=True,
    )

    assert tools == ()
    assert request.response_schema == schema
    assert request.metadata["mmm_output_token_ceiling"] == 333
    assert request.metadata["mmm_force_non_thinking"] is True
    rendered_messages = "\n".join(
        str(message.get("content") or "")
        for message in request.messages
    )
    assert "private_transport_field" not in rendered_messages
    assert '"properties"' not in rendered_messages
    assert "host owns the JSON shape" in rendered_messages


def test_planner_native_tool_transport_is_fail_closed() -> None:
    router = _PlannerRouter()

    with pytest.raises(
        RuntimeError,
        match="FIXED_TEMPLATE_PLANNER_NATIVE_TOOL_FORBIDDEN",
    ):
        _generate_native_template_arguments(
            router,
            "planner",
            ({"role": "user", "content": "x"},),
            tool_name="forbidden",
            parameters=_SCHEMA,
            description="must not run",
        )

    assert router.tool_calls == 0


def _schema_value(schema):
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
        return [_schema_value(schema.get("items", {})) for _ in range(minimum)]
    if "null" in types:
        return None
    if "object" in types or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties", {})
        return {
            name: _schema_value(properties[name])
            for name in schema.get("required", [])
        }
    raise AssertionError(f"unsupported schema in fixture: {schema!r}")


class _WorksheetRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_tool_decision(self, *_args, **_kwargs):
        raise AssertionError("planner worksheet generation must never use native tools")

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        del messages
        self.calls.append(dict(kwargs))
        schema = kwargs["response_schema"]
        properties = schema["properties"]
        if "record_count" in properties:
            return json.dumps({"record_count": 2})

        result = {}
        for name in schema["required"]:
            field_schema = properties[name]
            if (
                field_schema.get("type") == "array"
                and isinstance(field_schema.get("items"), dict)
                and field_schema["items"].get("type") == "object"
            ):
                count = int(field_schema.get("minItems", 0) or 0)
                assert count == int(field_schema.get("maxItems", count))
                item_schema = field_schema["items"]
                result[name] = [
                    {
                        field: _schema_value(item_schema["properties"][field])
                        for field in item_schema["required"]
                    }
                    for _ in range(count)
                ]
            else:
                result[name] = _schema_value(field_schema)
        return json.dumps(result)


def test_field_page_cannot_run_without_host_fixed_cardinality() -> None:
    router = _WorksheetRouter()
    section = "behavior_contract"
    page = pack_section_concerns(section)[0]

    with pytest.raises(
        ValueError,
        match="requires a host-fixed record count",
    ):
        _generate_authored_chunk(
            router,
            "make the requested gameplay feature",
            page=_PlannerPageRequest(
                section=section,
                chunk_index=1,
                chunk_count=1,
                concerns=page,
                completed={},
                include_evidence=False,
                media_paths=(),
            ),
        )

    assert router.calls == []


def test_authored_planner_uses_tiny_count_then_fixed_single_field_page() -> None:
    router = _WorksheetRouter()
    section = "behavior_contract"
    page = pack_section_concerns(section)[0]
    concern = str(page[0])

    count = _generate_concern_record_count(
        router,
        "make the requested gameplay feature",
        section=section,
        concern=concern,
        completed={},
    )
    assert count == 2
    count_call = router.calls[-1]
    assert count_call["enable_tools"] is False
    assert count_call["force_non_thinking"] is True
    assert (
        count_call["output_token_ceiling"]
        == PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING
    )
    assert list(count_call["response_schema"]["properties"]) == ["record_count"]

    value = _generate_authored_chunk(
        router,
        "make the requested gameplay feature",
        page=_PlannerPageRequest(
            section=section,
            chunk_index=1,
            chunk_count=1,
            concerns=page,
            completed={},
            include_evidence=False,
            media_paths=(),
        ),
        record_counts={concern: count},
    )
    assert len(value[concern]) == 2

    page_call = router.calls[-1]
    assert page_call["enable_tools"] is False
    assert page_call["force_non_thinking"] is True
    concern_schema = page_call["response_schema"]["properties"][concern]
    expected_ceiling = planner_page_output_token_ceiling(
        page_call["response_schema"]
    )
    assert page_call["output_token_ceiling"] == expected_ceiling
    assert expected_ceiling < PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING
    assert concern_schema["minItems"] == 2
    assert concern_schema["maxItems"] == 2
    assert len(concern_schema["items"]["properties"]) == 1


def test_planner_page_budget_is_schema_derived_and_rejects_unbounded_numeric() -> None:
    bounded = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "maxLength": 16},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }

    ceiling = planner_page_output_token_ceiling(bounded)
    assert 64 <= ceiling < PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING

    with pytest.raises(ValueError, match="unbounded lexical output"):
        planner_page_output_token_ceiling(
            {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        )


    with pytest.raises(ValueError, match="exceeds the global atomic output bound"):
        planner_page_output_token_ceiling(
            {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "maxLength": 1024},
                },
                "required": ["value"],
                "additionalProperties": False,
            }
        )
