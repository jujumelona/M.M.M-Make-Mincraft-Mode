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
    _MUTATION_ACT_TOOLS,
    _READ_OBSERVE_TOOLS,
    _VERIFY_TOOLS,
    LoopPhase,
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
                    "operation": {"type": "string", "maxLength": 256},
                    "path": {"type": "string", "maxLength": 256},
                    "content": {"type": "string", "maxLength": 256},
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
                "properties": {"query": {"type": "string", "maxLength": 256}},
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


def _page_response(
    arguments: dict[str, object],
    *,
    name: str = "apply_source_edit",
    content: str = "",
) -> GenerationResponse:
    raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return GenerationResponse(
        content=content,
        tool_calls=(
            ToolCall(
                id="native-page",
                name=name,
                arguments=dict(arguments),
                raw_arguments=raw,
            ),
        ),
    )


def _request_tool_name(request: GenerationRequest) -> str:
    choice = request.tool_choice
    assert isinstance(choice, dict)
    function = choice.get("function")
    assert isinstance(function, dict)
    return str(function.get("name") or "")


def _page_arguments(request: GenerationRequest, values: dict[str, object]) -> dict[str, object]:
    assert len(request.tools) == 1
    function = request.tools[0]["function"]
    parameters = function["parameters"]
    return {name: values[name] for name in parameters.get("properties", {}) if name in values}


def _assert_argument_page(request: GenerationRequest) -> None:
    assert request.parallel_tool_calls is False
    assert len(request.tools) == 1
    assert request.tool_validation_schemas == request.tools
    assert request.response_format == "text"
    assert request.response_schema is None
    tool = request.tools[0]
    function = tool.get("function")
    assert isinstance(function, dict)
    assert function.get("name") == _request_tool_name(request)
    parameters = function.get("parameters")
    assert isinstance(parameters, dict)
    assert parameters.get("type") == "object"
    assert parameters.get("additionalProperties") is False

def test_host_selected_mutation_uses_only_argument_contract() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            args = _page_arguments(request, _valid_arguments())
            return _page_response(args, name=_request_tool_name(request))

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 2
    for page in adapter.requests:
        _assert_argument_page(page)
    assert [call.name for call in result.tool_calls] == ["apply_source_edit"]
    assert result.tool_calls[0].arguments == _valid_arguments()
    assert result.tool_calls[0].id.startswith("host_mutation_")


def test_invalid_arguments_receive_one_same_contract_repair() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            name = _request_tool_name(request)
            if len(self.requests) == 1:
                return _page_response({}, name=name)
            return _page_response(
                _page_arguments(request, _valid_arguments()),
                name=name,
            )

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 3
    for request in adapter.requests:
        _assert_argument_page(request)
    assert "Repair the function arguments only" in adapter.requests[1].messages[-1]["content"]
    assert [call.name for call in result.tool_calls] == ["apply_source_edit"]


def test_repeated_invalid_argument_page_is_a_fixed_point() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return _page_response({}, name=_request_tool_name(request))

    _install_adapter_class(
        Adapter,
        transport_name="Regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    with pytest.raises(ModelConfigurationError, match="fixed point"):
        adapter.generate_turn(_mutation_request())

    assert len(adapter.requests) == 2
    for request in adapter.requests:
        _assert_argument_page(request)


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
    for request in adapter.requests:
        _assert_argument_page(request)


def test_host_selected_nonmutation_uses_same_argument_contract_without_probe() -> None:
    target = "java_workspace_symbols"

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return _page_response(
                {"query": "workspace"},
                name=_request_tool_name(request),
            )

    _install_adapter_class(
        Adapter,
        transport_name="Local regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_forced_query_request(target))

    assert len(adapter.requests) == 1
    _assert_argument_page(adapter.requests[0])
    assert [call.name for call in result.tool_calls] == [target]
    assert result.tool_calls[0].id.startswith("host_action_")


def test_malformed_message_json_is_irrelevant_when_forced_arguments_are_valid() -> None:
    target = "java_workspace_symbols"

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            return _page_response(
                {"query": "workspace"},
                name=_request_tool_name(request),
                content='{broken json: this must never be parsed',
            )

    _install_adapter_class(
        Adapter,
        transport_name="Local regression model",
        deterministic_stale_read=False,
    )
    adapter = Adapter()

    result = adapter.generate_turn(_forced_query_request(target))

    assert result.tool_calls[0].arguments == {"query": "workspace"}
    _assert_argument_page(adapter.requests[0])


def test_native_argument_recovery_contains_no_structured_text_transport() -> None:
    import inspect
    from minecraft_mod_ai import native_atomic_argument_recovery as recovery

    source = inspect.getsource(recovery)
    assert "json.loads(" not in source
    assert 'response_format="json"' not in source
    assert 'response_format="text"' in source
    assert "tool_choice={" in source

def test_host_tool_phase_classification_is_canonical() -> None:
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
    assert _MUTATION_ACT_TOOLS.isdisjoint(_READ_OBSERVE_TOOLS)
    assert _MUTATION_ACT_TOOLS.isdisjoint(_VERIFY_TOOLS)


def test_loop_phase_values_cover_all_execution_phases() -> None:
    phases = {p.value for p in LoopPhase}
    assert "OBSERVE" in phases
    assert "ACT" in phases
    assert "VERIFY" in phases
    assert "RECOVER" in phases
