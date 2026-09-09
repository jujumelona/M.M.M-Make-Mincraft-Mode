from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
)
from minecraft_mod_ai.model_output_atomicity_contract import (
    assert_atomic_model_schema,
    assert_installed,
    install,
    is_atomic_model_schema,
)


def _large_request(name: str, *, field_count: int = 12) -> GenerationRequest:
    properties = {f"field_{index}": {"type": "string"} for index in range(field_count)}
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "host-owned container",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
    return GenerationRequest(
        messages=({"role": "user", "content": "perform the already-selected action"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={"type": "function", "function": {"name": name}},
        response_format="text",
    )


def _valid_page_response(request: GenerationRequest) -> GenerationResponse:
    assert request.response_format == "json"
    assert isinstance(request.response_schema, dict)
    assert request.tools == ()
    assert request.tool_validation_schemas == ()
    assert request.tool_choice is None
    page_properties = request.response_schema["properties"]
    assert len(page_properties) <= 4
    return GenerationResponse(
        content=json.dumps(
            {name: f"value-{name}" for name in page_properties},
            sort_keys=True,
        )
    )


def test_large_closed_model_schema_is_allowed() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {f"nested_{inner}": {"type": "string"} for inner in range(4)},
                "additionalProperties": False,
            }
            for index in range(20)
        },
        "additionalProperties": False,
    }
    assert_atomic_model_schema(schema, surface="regression")
    assert is_atomic_model_schema(schema)


def test_small_atomic_schema_remains_allowed() -> None:
    assert_atomic_model_schema(
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        surface="regression",
    )


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
                    f"nested_{inner}": {"type": "string"}
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
        tool_name="oversized_planner_contract",
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
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    assert result == {"value": "ok"}
    assert calls == ["bounded_planner_contract"]


def test_legacy_raw_json_argument_recovery_helpers_are_removed() -> None:
    assert not hasattr(forced, "_argument_page_request")
    assert not hasattr(forced, "_argument_attempt")
    assert not hasattr(forced, "_argument_failure")


def test_large_host_owned_argument_container_is_decomposed_into_bounded_json_pages() -> None:
    request = _large_request("large_host_action")
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request)

    response = forced.host_selected_argument_turn(
        current,
        object(),
        request,
        "large_host_action",
    )

    assert len(observed) == 3
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "large_host_action"
    assert response.tool_calls[0].arguments == {
        f"field_{index}": f"value-field_{index}" for index in range(12)
    }
    assert all(turn.response_format == "json" for turn in observed)
    assert all(isinstance(turn.response_schema, dict) for turn in observed)
    assert all(turn.tools == () for turn in observed)


def test_mutation_recovery_uses_the_same_bounded_argument_only_json_pages() -> None:
    request = _large_request("generic_mutation_action", field_count=9)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request)

    response = forced.host_selected_mutation_turn(
        current,
        object(),
        request,
        "generic_mutation_action",
    )

    assert len(observed) == 3
    assert response.tool_calls[0].name == "generic_mutation_action"
    assert response.tool_calls[0].id.startswith("host_mutation_")
    assert all(turn.response_format == "json" for turn in observed)
    assert all(isinstance(turn.response_schema, dict) for turn in observed)
    assert all(turn.tools == () for turn in observed)


def test_invalid_json_page_repair_stays_argument_only_and_schema_bounded() -> None:
    request = _large_request("repairable_action", field_count=4)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        if len(observed) == 1:
            return GenerationResponse(content="not-json")
        return _valid_page_response(page_request)

    response = forced.host_selected_argument_turn(
        current,
        object(),
        request,
        "repairable_action",
    )

    assert len(observed) == 2
    assert response.tool_calls[0].name == "repairable_action"
    assert all(turn.response_format == "json" for turn in observed)
    assert all(isinstance(turn.response_schema, dict) for turn in observed)
    assert all(turn.tools == () for turn in observed)
    assert "Repair the arguments only" in observed[1].messages[-1]["content"]


def test_large_nested_field_reaches_generation_and_preserves_arguments() -> None:
    nested_properties = {
        f"nested_{index}": {"type": "string"} for index in range(40)
    }
    schema = {
        "type": "function",
        "function": {
            "name": "oversized_nested_action",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": nested_properties,
                        "required": list(nested_properties),
                        "additionalProperties": False,
                    }
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "perform the fixed action"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={"type": "function", "function": {"name": "oversized_nested_action"}},
        response_format="text",
    )
    calls = 0

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        nonlocal calls
        calls += 1
        assert_atomic_model_schema(page_request.response_schema, surface="nested field")
        return GenerationResponse(content=json.dumps({
            "payload": {key: "value" for key in nested_properties}
        }))

    response = forced.host_selected_argument_turn(
        current, object(), request, "oversized_nested_action",
    )
    assert calls == 1
    assert response.tool_calls[0].arguments == {
        "payload": {key: "value" for key in nested_properties}
    }


@pytest.mark.parametrize("kind", ["depth", "chars", "nodes", "properties"])
def test_schema_metadata_does_not_limit_model_generation(kind):
    schema = {
        "type": "object", "properties": {"value": {"type": "string"}},
        "additionalProperties": False,
    }
    if kind == "depth":
        for _ in range(12):
            schema = {"type": "object", "properties": {"child": schema},
                      "additionalProperties": False}
    elif kind == "chars":
        schema["description"] = "x" * 13000
    elif kind == "nodes":
        schema["properties"]["value"]["enum"] = [str(i) for i in range(150)]
    else:
        schema["properties"] = {f"field_{i}": {"type": "string"} for i in range(40)}
    observed = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            observed.append(kwargs["response_schema"])
            return "{}"

        def generate_tool_decision(self, *args, **kwargs):
            raise AssertionError("unexpected tool call")

    install(model_router_module=SimpleNamespace(ModelRouter=Router))
    Router().generate_text("planner", [], response_format="json", response_schema=schema)
    assert observed == [schema]
    assert is_atomic_model_schema(schema)


def test_open_object_template_still_rejected():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    with pytest.raises(ModelConfigurationError, match="MODEL_JSON_TEMPLATE_REQUIRED"):
        assert_atomic_model_schema(schema, surface="open template")
    assert not is_atomic_model_schema(schema)
