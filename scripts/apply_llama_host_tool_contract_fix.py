from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one patch anchor, found {text.count(old)}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Product: tool-capable structured calls may legitimately finish with the caller's
# JSON object without invoking a tool. Accept that as final JSON, but never interpret
# a malformed tool-shaped object as a final response.
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    """                return _parse_host_tool_envelope(raw, allowed, reasoning=reasoning)\n""",
    """                return _parse_host_tool_envelope(\n                    raw,\n                    allowed,\n                    allow_direct_json_final=request.response_format == \"json\",\n                    reasoning=reasoning,\n                )\n""",
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    """def _parse_host_tool_envelope(\n    raw: str,\n    allowed_tools: frozenset[str],\n    *,\n    reasoning: str = \"\",\n) -> GenerationResponse:\n""",
    """def _parse_host_tool_envelope(\n    raw: str,\n    allowed_tools: frozenset[str],\n    *,\n    allow_direct_json_final: bool = False,\n    reasoning: str = \"\",\n) -> GenerationResponse:\n""",
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    """    if kind != \"tool_calls\":\n        raise RuntimeError(\"Host tool envelope kind must be 'tool_calls' or 'final'\")\n\n    raw_calls = value.get(\"calls\")\n""",
    """    if kind != \"tool_calls\":\n        if allow_direct_json_final and \"calls\" not in value and \"tool_calls\" not in value:\n            # Tool availability does not force tool use. Preserve the caller's exact\n            # structured JSON so the normal host schema validator remains authoritative.\n            return GenerationResponse(content=raw.strip(), reasoning_content=reasoning)\n        raise RuntimeError(\"Host tool envelope kind must be 'tool_calls' or 'final'\")\n\n    raw_calls = value.get(\"calls\")\n""",
)

# Product: make accidental native llama.cpp tool transport impossible anywhere that
# calls the canonical payload builder directly. The adapter must translate tools to
# the host-owned envelope before constructing the server payload.
replace_once(
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    """    Tool-capable turns use the smallest widely compatible function-calling wire\n    contract. Structured non-tool turns deliberately do *not* send response_format,\n    json_schema, or grammar to llama.cpp: those controls can compile through fragile\n    server-side GBNF and fail before the model runs. JSON syntax/schema validation and\n    isolated repair are host-owned. We only disable model-internal thinking here.\n""",
    """    Native tool/JSON sampler controls are never sent to llama.cpp. They can compile\n    through server-side GBNF and fail before the model runs. Tool-capable turns must\n    first be translated by LlamaCppAdapter into the host-owned JSON envelope; direct\n    tool metadata here is therefore a programming error. JSON syntax/schema validation\n    and isolated repair are host-owned. We only disable model-internal thinking here.\n""",
)
replace_once(
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    """    tools = getattr(request, \"tools\", ()) or ()\n    if tools:\n        payload[\"tools\"] = [dict(tool) for tool in tools]\n        tool_choice = getattr(request, \"tool_choice\", None)\n        if tool_choice is not None:\n            payload[\"tool_choice\"] = (\n                dict(tool_choice) if isinstance(tool_choice, dict) else tool_choice\n            )\n        # Do not combine optional structured/reasoning/parallel transport controls\n        # with native tool calls. Some llama-server/chat-template combinations reject\n        # that mixed payload even though each feature is valid independently.\n        return payload\n\n""",
    """    tools = getattr(request, \"tools\", ()) or ()\n    if tools:\n        raise RuntimeError(\n            \"Native llama-server tool transport is disabled; translate tools through \"\n            \"the host-owned tool envelope before building the server payload\"\n        )\n\n""",
)

# Regression test: direct final JSON is valid even when tools were available, and the
# actual HTTP body must contain no native grammar/tool controls.
replace_once(
    "tests/test_llama_cpp_adapter_request_contract.py",
    '''def test_generate_turn_reuses_canonical_payload_for_json_tools(monkeypatch) -> None:\n    captured: dict[str, object] = {}\n    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")\n    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())\n\n    def post(url, *, json, timeout):\n        captured["url"] = url\n        captured["payload"] = json\n        return _CompletionResponse(\n            status_code=200,\n            payload={"choices": [{"message": {"content": '{"game_design":{}}'}}]},\n        )\n\n    monkeypatch.setattr(httpx, "post", post)\n    request = GenerationRequest(\n        messages=({"role": "user", "content": "plan"},),\n        response_format="json",\n        tools=(\n            {\n                "type": "function",\n                "function": {\n                    "name": "lookup",\n                    "description": "lookup",\n                    "parameters": {"type": "object", "properties": {}},\n                },\n            },\n        ),\n        tool_choice="auto",\n        parallel_tool_calls=True,\n    )\n\n    turn = _adapter().generate_turn(request)\n\n    assert turn.content == '{"game_design":{}}'\n    payload = captured["payload"]\n    assert payload["tools"][0]["function"]["name"] == "lookup"\n    assert payload["tool_choice"] == "auto"\n    assert "response_format" not in payload\n    assert "reasoning_effort" not in payload\n    assert "parallel_tool_calls" not in payload\n''',
    '''def test_generate_turn_accepts_direct_final_json_without_native_tool_transport(monkeypatch) -> None:\n    captured: dict[str, object] = {}\n    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")\n    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())\n\n    def post(url, *, json, timeout):\n        captured["url"] = url\n        captured["payload"] = json\n        return _CompletionResponse(\n            status_code=200,\n            payload={"choices": [{"message": {"content": '{"game_design":{}}'}}]},\n        )\n\n    monkeypatch.setattr(httpx, "post", post)\n    request = GenerationRequest(\n        messages=({"role": "user", "content": "plan"},),\n        response_format="json",\n        tools=(\n            {\n                "type": "function",\n                "function": {\n                    "name": "lookup",\n                    "description": "lookup",\n                    "parameters": {"type": "object", "properties": {}},\n                },\n            },\n        ),\n        tool_choice="auto",\n        parallel_tool_calls=True,\n    )\n\n    turn = _adapter().generate_turn(request)\n\n    assert turn.content == '{"game_design":{}}'\n    payload = captured["payload"]\n    for forbidden in (\n        "tools",\n        "tool_choice",\n        "parallel_tool_calls",\n        "response_format",\n        "json_schema",\n        "grammar",\n    ):\n        assert forbidden not in payload\n    rendered_messages = "\\n".join(str(message.get("content", "")) for message in payload["messages"])\n    assert "REVIEWED_TOOL_CATALOG" in rendered_messages\n    assert "lookup" in rendered_messages\n\n\ndef test_generate_turn_parses_host_tool_envelope_without_native_tool_fields(monkeypatch) -> None:\n    captured: dict[str, object] = {}\n    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")\n    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())\n\n    def post(url, *, json, timeout):\n        captured["payload"] = json\n        return _CompletionResponse(\n            status_code=200,\n            payload={\n                "choices": [\n                    {\n                        "message": {\n                            "content": (\n                                '{"kind":"tool_calls","calls":['\n                                '{"id":"call_7","name":"lookup","arguments":{"q":"x"}}]}'\n                            )\n                        }\n                    }\n                ]\n            },\n        )\n\n    monkeypatch.setattr(httpx, "post", post)\n    request = GenerationRequest(\n        messages=({"role": "user", "content": "look it up"},),\n        response_format="json",\n        tools=(\n            {\n                "type": "function",\n                "function": {\n                    "name": "lookup",\n                    "description": "lookup",\n                    "parameters": {"type": "object", "properties": {}},\n                },\n            },\n        ),\n        tool_choice="auto",\n        parallel_tool_calls=True,\n    )\n\n    turn = _adapter().generate_turn(request)\n    assert turn.content == ""\n    assert len(turn.tool_calls) == 1\n    assert turn.tool_calls[0].id == "call_7"\n    assert turn.tool_calls[0].name == "lookup"\n    assert turn.tool_calls[0].arguments == {"q": "x"}\n    for forbidden in ("tools", "tool_choice", "parallel_tool_calls", "response_format", "json_schema", "grammar"):\n        assert forbidden not in captured["payload"]\n''',
)

# Architectural tests: canonical server payload must reject native tool metadata rather
# than silently enabling llama.cpp's sampler grammar path.
replace_once(
    "tests/test_llama_structured_reasoning_policy.py",
    '''def test_tool_turn_keeps_minimal_function_calling_transport() -> None:\n    payload = llama_server_hardware_policy._server_payload(\n        _adapter(),\n        _request(\n            response_format="json",\n            tools=(\n                {\n                    "type": "function",\n                    "function": {\n                        "name": "lookup",\n                        "description": "lookup",\n                        "parameters": {"type": "object", "properties": {}},\n                    },\n                },\n            ),\n        ),\n    )\n\n    assert "tools" in payload\n    assert "response_format" not in payload\n    assert "reasoning_effort" not in payload\n    assert "chat_template_kwargs" not in payload\n''',
    '''def test_native_tool_transport_is_rejected_by_canonical_payload_builder() -> None:\n    try:\n        llama_server_hardware_policy._server_payload(\n            _adapter(),\n            _request(\n                response_format="json",\n                tools=(\n                    {\n                        "type": "function",\n                        "function": {\n                            "name": "lookup",\n                            "description": "lookup",\n                            "parameters": {"type": "object", "properties": {}},\n                        },\n                    },\n                ),\n            ),\n        )\n    except RuntimeError as exc:\n        assert "Native llama-server tool transport is disabled" in str(exc)\n    else:\n        raise AssertionError("native llama tool metadata must fail closed")\n''',
)

replace_once(
    "tests/test_runtime_json_gap_regression.py",
    '''def test_tool_json_turn_never_mixes_native_grammar_controls():\n    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))\n    tool = {\n        "type": "function",\n        "function": {\n            "name": "search_project_rag",\n            "description": "search",\n            "parameters": {"type": "object", "properties": {}},\n        },\n    }\n    request = SimpleNamespace(\n        messages=({"role": "user", "content": "use the tool"},),\n        tools=(tool,),\n        tool_choice="auto",\n        response_format="json",\n    )\n    payload = _server_payload(adapter, request)\n    assert payload["tools"] == [tool]\n    assert payload["tool_choice"] == "auto"\n    assert "response_format" not in payload\n    assert "json_schema" not in payload\n    assert "grammar" not in payload\n    assert "reasoning_effort" not in payload\n    assert "chat_template_kwargs" not in payload\n''',
    '''def test_tool_json_turn_cannot_reenter_native_llama_grammar_path():\n    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))\n    tool = {\n        "type": "function",\n        "function": {\n            "name": "search_project_rag",\n            "description": "search",\n            "parameters": {"type": "object", "properties": {}},\n        },\n    }\n    request = SimpleNamespace(\n        messages=({"role": "user", "content": "use the tool"},),\n        tools=(tool,),\n        tool_choice="auto",\n        response_format="json",\n    )\n    try:\n        _server_payload(adapter, request)\n    except RuntimeError as exc:\n        assert "Native llama-server tool transport is disabled" in str(exc)\n    else:\n        raise AssertionError("native llama tool metadata must fail closed")\n''',
)

print("host-tool contract patch applied")
