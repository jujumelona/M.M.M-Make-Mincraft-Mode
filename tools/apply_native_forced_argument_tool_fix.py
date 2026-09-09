from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "minecraft_mod_ai" / "native_atomic_argument_recovery.py"
TEST = ROOT / "tests" / "test_host_selected_mutation_action.py"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


def patch_native() -> None:
    text = NATIVE.read_text(encoding="utf-8")
    text = text.replace(
        "therefore not ask a small model to select a tool again. Each bounded page is requested as\n"
        "a JSON object constrained by the page's argument schema, model-returned tool calls are\n"
        "ignored, and the host constructs the final ToolCall only after every page validates.",
        "therefore not ask a small model to select a semantic action again. Each bounded page is\n"
        "exposed as exactly one forced native function whose parameters are the page schema. The\n"
        "host consumes only the returned ToolCall.arguments mapping, merges validated pages, and\n"
        "constructs the final ToolCall. Model-authored JSON text is never parsed on this path.",
    )

    messages = '''def _messages(
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    action_name: str,
    repair_error: str = "",
) -> tuple[dict[str, Any], ...]:
    messages = [
        dict(raw)
        for raw in tuple(getattr(request, "messages", ()) or ())
        if isinstance(raw, Mapping)
    ]
    properties = page_schema.get("properties")
    fields = (
        ", ".join(str(name) for name in properties)
        if isinstance(properties, Mapping)
        else "arguments"
    )
    instruction = (
        f"The host already selected action {action_name!r}. "
        f"Call that required function exactly once for argument page {page_index}/{page_count}. "
        f"Fill only these argument fields: {fields}. "
        "Do not answer in prose and do not serialize a JSON object into message content. "
        "The host owns action selection, merges bounded pages, validates the complete object, "
        "and constructs the final executable tool call."
    )
    if repair_error:
        instruction += (
            " Repair the function arguments only. The previous forced-call arguments were invalid. "
            "Validation: " + repair_error[:_MAX_REPAIR_ERROR_CHARS]
        )
    messages.append({"role": "user", "content": instruction})
    return tuple(messages)
'''
    text = replace_between(text, "def _messages(\n", "def _request(\n", messages)

    request = '''def _request(
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    action_name: str,
    repair_error: str = "",
) -> Any:
    # The atomic boundary is checked per native function-argument page, never against
    # the host-owned original container. The model fills fields through ToolCall.arguments;
    # message content is deliberately not a structured-output transport.
    from .model_output_atomicity_contract import assert_atomic_model_schema

    assert_atomic_model_schema(
        page_schema,
        surface="host-selected forced-function argument page",
    )
    page_tool = {
        "type": "function",
        "function": {
            "name": action_name,
            "description": (
                f"Fill host-selected argument page {page_index}/{page_count}. "
                "Return values only through function arguments."
            ),
            "parameters": dict(page_schema),
        },
    }
    return replace(
        request,
        messages=_messages(
            request,
            page_index=page_index,
            page_count=page_count,
            page_schema=page_schema,
            action_name=action_name,
            repair_error=repair_error,
        ),
        tools=(page_tool,),
        tool_validation_schemas=(page_tool,),
        tool_choice={"type": "function", "function": {"name": action_name}},
        parallel_tool_calls=False,
        response_format="text",
        response_schema=None,
    )
'''
    text = replace_between(text, "def _request(\n", "def _fingerprint(\n", request)

    page_result = '''def _page_result(
    turn: Any,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
    action_name: str,
) -> tuple[dict[str, Any] | None, str, str]:
    """Consume only one native forced ToolCall; message content is never parsed as JSON."""

    forced = _forced_module()
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    matches = tuple(call for call in calls if str(getattr(call, "name", "")) == action_name)
    if len(calls) != 1 or len(matches) != 1:
        reason = (
            f"argument page must return exactly one forced {action_name!r} tool call; "
            f"received {len(calls)} tool call(s)"
        )
        return None, reason, _fingerprint(
            {
                "tool_calls": [
                    {
                        "name": str(getattr(call, "name", "")),
                        "arguments": getattr(call, "arguments", None),
                    }
                    for call in calls
                ]
            }
        )
    raw_arguments = getattr(matches[0], "arguments", None)
    if not isinstance(raw_arguments, Mapping):
        reason = "forced argument page tool call did not expose an arguments object"
        return None, reason, _fingerprint({"arguments": raw_arguments})
    normalized = _page_owned_arguments(dict(raw_arguments), page_schema, parameters)
    if not forced._arguments_match_schema(normalized, page_schema):
        diag = getattr(forced, "_schema_validation_diagnostics", lambda *args: "")(
            normalized, page_schema
        )
        reason = (
            f"forced argument page failed the host page schema ({diag})"
            if diag
            else "forced argument page failed the host page schema"
        )
        return None, reason, _fingerprint(normalized)
    return normalized, "", _fingerprint(normalized)
'''
    text = replace_between(text, "def _page_result(\n", "def _page_attempt(\n", page_result)

    page_attempt = '''def _page_attempt(
    current: Any,
    adapter: Any,
    request: Any,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
    action_name: str,
) -> tuple[dict[str, Any] | None, str, str]:
    try:
        turn = current(adapter, request)
    except Exception as exc:
        from .llama_finish_reason_contract import completion_boundary_error
        from .generation_output_budget import GenerationOutputBudgetError

        # Backend/context failures belong to the canonical recovery owner. Retrying
        # them as invalid arguments loses their type, cause and preserved partial receipt.
        if completion_boundary_error(exc) is not None or isinstance(exc, GenerationOutputBudgetError):
            raise
        cause = getattr(exc, "cause", exc)
        reason = f"{type(cause).__name__}: {cause}"[:_MAX_REPAIR_ERROR_CHARS]
        return None, reason, _fingerprint({"exception": reason})
    return _page_result(turn, page_schema, parameters, action_name)
'''
    text = replace_between(text, "def _page_attempt(\n", "def _recover_page(\n", page_attempt)

    recover_page = '''def _recover_page(
    current: Any,
    adapter: Any,
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
    action_name: str,
) -> dict[str, Any]:
    from .model_adapters import ModelConfigurationError

    first_request = _request(
        request,
        page_index=page_index,
        page_count=page_count,
        page_schema=page_schema,
        action_name=action_name,
    )
    arguments, error, first_fingerprint = _page_attempt(
        current,
        adapter,
        first_request,
        page_schema,
        parameters,
        action_name,
    )
    if arguments is not None:
        return arguments

    repair_request = _request(
        request,
        page_index=page_index,
        page_count=page_count,
        page_schema=page_schema,
        action_name=action_name,
        repair_error=error,
    )
    arguments, repair_error, second_fingerprint = _page_attempt(
        current,
        adapter,
        repair_request,
        page_schema,
        parameters,
        action_name,
    )
    if arguments is not None:
        return arguments

    fixed_point = first_fingerprint == second_fingerprint
    suffix = (
        "repeated-invalid forced argument-page fixed point"
        if fixed_point
        else "bounded forced argument-page repair exhausted"
    )
    raise ModelConfigurationError(
        f"Host-selected action {action_name!r} {suffix} on page "
        f"{page_index}/{page_count}; error={repair_error or error}."
    )
'''
    text = replace_between(text, "def _recover_page(\n", "def _recover_source_edit_arguments(\n", recover_page)

    text = text.replace(
        '"""Recover one already-selected action through bounded argument-only JSON pages."""',
        '"""Recover one already-selected action through bounded forced-function pages."""',
    )

    forbidden = [
        'json.loads(',
        'response_format="json"',
        'tools=(),\n        tool_validation_schemas=(),\n        tool_choice=None',
    ]
    leftovers = [needle for needle in forbidden if needle in text]
    if leftovers:
        raise SystemExit(f"unsafe native structured-text transport remains: {leftovers}")
    NATIVE.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TEST.read_text(encoding="utf-8")

    helpers = '''def _page_response(
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
'''
    text = replace_between(text, "def _page_response(\n", "def test_host_selected_mutation_uses_only_argument_contract()", helpers)

    first_tests = '''def test_host_selected_mutation_uses_only_argument_contract() -> None:
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
    assert "Do not serialize a JSON object" in source
'''
    text = replace_between(text, "def test_host_selected_mutation_uses_only_argument_contract()", "def test_host_tool_phase_classification_is_canonical()", first_tests)

    TEST.write_text(text, encoding="utf-8")


def main() -> None:
    patch_native()
    patch_tests()
    print("native forced-argument transport migrated to native tool calls")


if __name__ == "__main__":
    main()
