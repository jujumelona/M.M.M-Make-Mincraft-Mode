from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.forced_tool_execution_contract import _install_adapter_class
from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    LoopPhase,
    _MUTATION_ACT_TOOLS,
    _READ_OBSERVE_TOOLS,
    _VERIFY_TOOLS,
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


def test_host_tool_phase_classification_is_canonical() -> None:
    """Tool → phase mapping is host-owned via the canonical tool sets."""
    assert "search_code_rag" in _READ_OBSERVE_TOOLS
    assert "search_project_rag" in _READ_OBSERVE_TOOLS
    assert "external_mcp_call" in _READ_OBSERVE_TOOLS
    assert "java_workspace_symbols" in _READ_OBSERVE_TOOLS

    assert "apply_source_edit" in _MUTATION_ACT_TOOLS
    assert "apply_source_patch" in _MUTATION_ACT_TOOLS
    assert "apply_java_operations" in _MUTATION_ACT_TOOLS

    assert "java_diagnostics" in _VERIFY_TOOLS
    assert "run_gradle_build" in _VERIFY_TOOLS
    assert "run_gametest" in _VERIFY_TOOLS


def test_mutation_tool_set_is_disjoint_from_observe_and_verify() -> None:
    """Mutation tools must not overlap with observe or verify sets."""
    assert _MUTATION_ACT_TOOLS.isdisjoint(_READ_OBSERVE_TOOLS)
    assert _MUTATION_ACT_TOOLS.isdisjoint(_VERIFY_TOOLS)


def test_loop_phase_values_cover_all_execution_phases() -> None:
    """LoopPhase enum must cover the four canonical execution phases."""
    phases = {p.value for p in LoopPhase}
    assert "OBSERVE" in phases
    assert "ACT" in phases
    assert "VERIFY" in phases
    assert "RECOVER" in phases

