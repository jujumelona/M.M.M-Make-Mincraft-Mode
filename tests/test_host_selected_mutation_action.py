from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.causal_action_state import (
    CausalAction,
    classify_action,
    select_action_frontier,
)
from minecraft_mod_ai.forced_tool_execution_contract import _install_adapter_class
from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["operation", "path", "content"],
            },
        },
    }


def _query_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def _mutation_request() -> GenerationRequest:
    edit = _schema("apply_source_edit")
    stale = _query_schema("java_workspace_symbols")
    return GenerationRequest(
        messages=({"role": "user", "content": "repair the source"},),
        tools=(edit,),
        tool_validation_schemas=(edit, stale),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )


def _forced_query_request(name: str) -> GenerationRequest:
    schema = _query_schema(name)
    return GenerationRequest(
        messages=({"role": "user", "content": "inspect the workspace"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={"type": "function", "function": {"name": name}},
        parallel_tool_calls=False,
    )


def _valid_arguments() -> dict[str, str]:
    return {
        "operation": "create_file",
        "path": "src/main/java/example/Fixed.java",
        "content": "package example; final class Fixed {}",
    }


def test_host_selected_mutation_never_exposes_tool_name_to_model() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return GenerationResponse(content=json.dumps(_valid_arguments()))

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 1
    page = adapter.requests[0]
    assert page.tools == ()
    assert page.tool_validation_schemas == ()
    assert page.tool_choice is None
    assert page.parallel_tool_calls is False
    assert page.response_format == "json"
    assert page.response_schema == _schema("apply_source_edit")["function"]["parameters"]
    assert [call.name for call in result.tool_calls] == ["apply_source_edit"]
    assert result.tool_calls[0].arguments == _valid_arguments()
    assert result.tool_calls[0].id.startswith("host_mutation_")


def test_invalid_arguments_receive_one_argument_only_repair() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return GenerationResponse(content="{}")
            return GenerationResponse(content=json.dumps(_valid_arguments()))

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 2
    assert all(request.tools == () for request in adapter.requests)
    assert all(request.tool_choice is None for request in adapter.requests)
    assert "Repair the arguments only" in adapter.requests[1].messages[-1]["content"]
    assert [call.name for call in result.tool_calls] == ["apply_source_edit"]


def test_repeated_invalid_argument_page_is_a_fixed_point() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return GenerationResponse(content="{}")

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    with pytest.raises(ModelConfigurationError, match="fixed point"):
        adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 2
    assert all(request.tools == () for request in adapter.requests)


def test_argument_page_never_executes_stale_tool_call() -> None:
    stale = ToolCall(
        id="stale-call",
        name="java_workspace_symbols",
        arguments={"query": "old"},
        raw_arguments='{"query":"old"}',
    )

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return GenerationResponse(tool_calls=(stale,))

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    with pytest.raises(ModelConfigurationError, match="fixed point"):
        adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 2
    assert all(request.tools == () for request in adapter.requests)
    assert all(request.tool_validation_schemas == () for request in adapter.requests)


def test_failed_native_required_probe_falls_back_to_argument_page() -> None:
    target = "java_workspace_symbols"

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def _server_url(self, request: GenerationRequest) -> str:
            return "http://probe-fallback.test"

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return GenerationResponse(content="native required was not enforced")
            return GenerationResponse(content='{"query":"workspace"}')

    _install_adapter_class(
        Adapter,
        transport_name="Local regression model",
        deterministic_stale_read=False,
        probe_native_required=True,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_forced_query_request(target))

    assert len(adapter.requests) == 2
    probe, fallback = adapter.requests
    assert probe.tool_choice == "required"
    assert probe.tools[0]["function"]["name"] == "mmm_required_tool_probe"
    assert fallback.tools == ()
    assert fallback.tool_choice is None
    assert fallback.response_format == "json"
    assert [call.name for call in result.tool_calls] == [target]
    assert result.tool_calls[0].id.startswith("host_action_")


def test_successful_native_required_probe_uses_native_action_once() -> None:
    target = "java_workspace_symbols"

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def _server_url(self, request: GenerationRequest) -> str:
            return "http://probe-native.test"

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="probe",
                            name="mmm_required_tool_probe",
                            arguments={"nonce": "mmm"},
                            raw_arguments='{"nonce":"mmm"}',
                        ),
                    )
                )
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="native",
                        name=target,
                        arguments={"query": "workspace"},
                        raw_arguments='{"query":"workspace"}',
                    ),
                )
            )

    _install_adapter_class(
        Adapter,
        transport_name="Local regression model",
        deterministic_stale_read=False,
        probe_native_required=True,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_forced_query_request(target))

    assert len(adapter.requests) == 2
    assert adapter.requests[1].tool_choice == "required"
    assert [item["function"]["name"] for item in adapter.requests[1].tools] == [target]
    assert [call.name for call in result.tool_calls] == [target]
    assert result.tool_calls[0].id == "native"


def test_causal_action_class_is_host_derived() -> None:
    assert classify_action(()) is CausalAction.FINISH
    assert classify_action(("search_code_rag",)) is CausalAction.RETRIEVE
    assert classify_action(("java_workspace_symbols",)) is CausalAction.INSPECT
    assert classify_action(("apply_source_edit",)) is CausalAction.MUTATE
    assert classify_action(("run_gradle_build",)) is CausalAction.VERIFY


def test_ranked_frontier_selects_action_class_before_tool_surface() -> None:
    action, names = select_action_frontier(
        ("search_code_rag", "java_workspace_symbols", "search_project_rag")
    )
    assert action is CausalAction.RETRIEVE
    assert names == ("search_code_rag", "search_project_rag")


def test_mutation_frontier_collapses_to_one_host_tool() -> None:
    action, names = select_action_frontier(
        ("apply_source_edit", "apply_source_patch", "java_workspace_symbols")
    )
    assert action is CausalAction.MUTATE
    assert names == ("apply_source_edit",)
