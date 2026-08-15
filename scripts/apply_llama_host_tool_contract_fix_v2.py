from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one patch anchor, found {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# A tool-capable structured request may legitimately finish without using a tool.
# Accept a caller-shaped JSON object as the final structured response, while keeping
# anything tool-shaped behind the explicit, validated host tool envelope.
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    "                return _parse_host_tool_envelope(raw, allowed, reasoning=reasoning)\n",
    """                return _parse_host_tool_envelope(
                    raw,
                    allowed,
                    allow_direct_json_final=request.response_format == "json",
                    reasoning=reasoning,
                )
""",
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    """def _parse_host_tool_envelope(
    raw: str,
    allowed_tools: frozenset[str],
    *,
    reasoning: str = "",
) -> GenerationResponse:
""",
    """def _parse_host_tool_envelope(
    raw: str,
    allowed_tools: frozenset[str],
    *,
    allow_direct_json_final: bool = False,
    reasoning: str = "",
) -> GenerationResponse:
""",
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    """    if kind != "tool_calls":
        raise RuntimeError("Host tool envelope kind must be 'tool_calls' or 'final'")

    raw_calls = value.get("calls")
""",
    """    if kind != "tool_calls":
        if allow_direct_json_final and "calls" not in value and "tool_calls" not in value:
            # Tool availability does not force tool use. Preserve the caller's exact
            # structured JSON so the normal host schema validator remains authoritative.
            return GenerationResponse(content=raw.strip(), reasoning_content=reasoning)
        raise RuntimeError("Host tool envelope kind must be 'tool_calls' or 'final'")

    raw_calls = value.get("calls")
""",
)

# Make the canonical server payload fail closed if any caller tries to bypass the
# host-owned envelope and send native llama.cpp tool metadata again.
replace_once(
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    """    Tool-capable turns use the smallest widely compatible function-calling wire
    contract. Structured non-tool turns deliberately do *not* send response_format,
    json_schema, or grammar to llama.cpp: those controls can compile through fragile
    server-side GBNF and fail before the model runs. JSON syntax/schema validation and
    isolated repair are host-owned. We only disable model-internal thinking here.
""",
    """    Native tool/JSON sampler controls are never sent to llama.cpp. They can compile
    through server-side GBNF and fail before the model runs. Tool-capable turns must
    first be translated by LlamaCppAdapter into the host-owned JSON envelope; direct
    tool metadata here is therefore a programming error. JSON syntax/schema validation
    and isolated repair are host-owned. We only disable model-internal thinking here.
""",
)
replace_once(
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    """    tools = getattr(request, "tools", ()) or ()
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is not None:
            payload["tool_choice"] = (
                dict(tool_choice) if isinstance(tool_choice, dict) else tool_choice
            )
        # Do not combine optional structured/reasoning/parallel transport controls
        # with native tool calls. Some llama-server/chat-template combinations reject
        # that mixed payload even though each feature is valid independently.
        return payload

""",
    """    tools = getattr(request, "tools", ()) or ()
    if tools:
        raise RuntimeError(
            "Native llama-server tool transport is disabled; translate tools through "
            "the host-owned tool envelope before building the server payload"
        )

""",
)

# Replace the stale adapter test with two production-contract tests: direct final JSON
# and actual host tool-call envelope. Both assert the HTTP body is grammar/tool free.
replace_once(
    "tests/test_llama_cpp_adapter_request_contract.py",
    '''def test_generate_turn_reuses_canonical_payload_for_json_tools(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"game_design":{}}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    request = GenerationRequest(
        messages=({"role": "user", "content": "plan"},),
        response_format="json",
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)

    assert turn.content == '{"game_design":{}}'
    payload = captured["payload"]
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tool_choice"] == "auto"
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "parallel_tool_calls" not in payload
''',
    '''def test_generate_turn_accepts_direct_final_json_without_native_tool_transport(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"game_design":{}}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    request = GenerationRequest(
        messages=({"role": "user", "content": "plan"},),
        response_format="json",
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)

    assert turn.content == '{"game_design":{}}'
    payload = captured["payload"]
    for forbidden in (
        "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "json_schema", "grammar",
    ):
        assert forbidden not in payload
    rendered = "\\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "REVIEWED_TOOL_CATALOG" in rendered
    assert "lookup" in rendered


def test_generate_turn_parses_host_tool_envelope_without_native_tool_fields(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={
                "choices": [{"message": {"content": (
                    '{"kind":"tool_calls","calls":['
                    '{"id":"call_7","name":"lookup","arguments":{"q":"x"}}]}'
                )}}]
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    request = GenerationRequest(
        messages=({"role": "user", "content": "look it up"},),
        response_format="json",
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _adapter().generate_turn(request)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_7"
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "x"}
    for forbidden in (
        "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "json_schema", "grammar",
    ):
        assert forbidden not in captured["payload"]
''',
)

replace_once(
    "tests/test_llama_structured_reasoning_policy.py",
    '''def test_tool_turn_keeps_minimal_function_calling_transport() -> None:
    payload = llama_server_hardware_policy._server_payload(
        _adapter(),
        _request(
            response_format="json",
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ),
        ),
    )

    assert "tools" in payload
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload
''',
    '''def test_native_tool_transport_is_rejected_by_canonical_payload_builder() -> None:
    try:
        llama_server_hardware_policy._server_payload(
            _adapter(),
            _request(
                response_format="json",
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ),
            ),
        )
    except RuntimeError as exc:
        assert "Native llama-server tool transport is disabled" in str(exc)
    else:
        raise AssertionError("native llama tool metadata must fail closed")
''',
)

print("corrected host-tool contract patch applied")
