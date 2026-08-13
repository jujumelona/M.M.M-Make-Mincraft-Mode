from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_function(path: str, name: str, replacement: str) -> None:
    target = ROOT / path
    source = target.read_text(encoding="utf-8")
    marker = f"def {name}("
    start = source.find(marker)
    if start < 0:
        raise SystemExit(f"{path}: function not found: {name}")
    if start > 0 and source[start - 1] not in {"\n", "\r"}:
        raise SystemExit(f"{path}: unexpected function anchor for {name}")
    next_def = source.find("\ndef ", start + len(marker))
    next_class = source.find("\nclass ", start + len(marker))
    ends = [value for value in (next_def, next_class) if value >= 0]
    end = min(ends) if ends else len(source)
    prefix = source[:start]
    suffix = source[end:]
    target.write_text(prefix + replacement.rstrip() + "\n" + suffix, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, got {count}: {old[:120]!r}")
    target.write_text(source.replace(old, new, 1), encoding="utf-8")


replace_function(
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    "_server_payload",
    '''def _server_payload(adapter: Any, request: Any) -> dict[str, Any]:
    """Build the one authoritative native llama-server chat payload.

    Tool-capable turns use the smallest widely compatible function-calling wire
    contract. Structured non-tool turns may use llama.cpp JSON-schema constrained
    decoding. Host parsing and validation remain authoritative in both cases.
    """

    payload: dict[str, Any] = {
        "model": "local",
        "messages": [dict(message) for message in request.messages],
        "max_tokens": int(adapter.config.max_new_tokens),
        "temperature": 0.0,
    }
    tools = getattr(request, "tools", ()) or ()
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

    if getattr(request, "response_format", None) == "json":
        schema = getattr(request, "response_schema", None)
        if schema is not None:
            payload["response_format"] = {
                "type": "json_object",
                "schema": dict(schema),
            }
        else:
            payload["response_format"] = {"type": "json_object"}
    return payload
''',
)

replace_once(
    "minecraft_mod_ai/game_design.py",
    '''            response_format="json",\n            response_schema=_GAME_DESIGN_RESPONSE_SCHEMA,\n        )\n''',
    '''            response_format="json",\n            response_schema=_GAME_DESIGN_RESPONSE_SCHEMA,\n            enable_tools=False,\n        )\n''',
)

# No later runtime wrapper may silently strip structured output again.
for rel in (
    "minecraft_mod_ai/llama_stream_efficiency_contract.py",
    "minecraft_mod_ai/llama_decode_speed_contract.py",
):
    source = (ROOT / rel).read_text(encoding="utf-8")
    if "_install_host_validated_json_payload" in source:
        raise SystemExit(f"{rel}: obsolete JSON payload stripping wrapper remains")

replace_function(
    "tests/test_llama_structured_reasoning_stream.py",
    "test_json_request_uses_host_validation_without_optional_transport_controls",
    '''def test_json_request_uses_schema_when_no_tools_are_present() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    request = SimpleNamespace(
        messages=(
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Plan it."},
        ),
        response_format="json",
        response_schema=schema,
        tools=(),
    )
    payload = _server_payload(_adapter(), request)
    assert payload["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }
    assert "reasoning_effort" not in payload
    assert "parallel_tool_calls" not in payload
    assert payload["max_tokens"] == 8192
''',
)

# Replace tool test so it proves schema is intentionally suppressed only for tool turns.
replace_function(
    "tests/test_llama_structured_reasoning_stream.py",
    "test_tool_request_uses_same_minimal_native_payload",
    '''def test_tool_request_uses_minimal_payload_even_when_schema_exists() -> None:
    request = SimpleNamespace(
        messages=({"role": "user", "content": "inspect then plan"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup evidence",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    payload = _server_payload(_adapter(), request)
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tool_choice"] == "auto"
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "parallel_tool_calls" not in payload
''',
)

# Existing direct-adapter tool test remains the compatibility proof. Add a no-tool
# adapter test to prove direct generate_turn shares the same schema-capable owner.
adapter_test = ROOT / "tests/test_llama_cpp_adapter_request_contract.py"
source = adapter_test.read_text(encoding="utf-8")
if "test_generate_turn_preserves_schema_for_non_tool_json" not in source:
    source = source.rstrip() + '''\n\n\ndef test_generate_turn_preserves_schema_for_non_tool_json(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())

    def post(url, *, json, timeout):
        captured["payload"] = json
        return _CompletionResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    turn = _adapter().generate_turn(
        GenerationRequest(
            messages=({"role": "user", "content": "structured"},),
            response_format="json",
            response_schema=schema,
        )
    )

    assert turn.content == '{"value":"ok"}'
    assert captured["payload"]["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }
''' + "\n"
    adapter_test.write_text(source, encoding="utf-8")

# Strengthen the game-design contract: it must not instantiate the agent-tool loop.
game_test = ROOT / "tests/test_game_design_router.py"
source = game_test.read_text(encoding="utf-8")
old = '''    schema = router.calls[0][1]["response_schema"]\n    assert schema["required"] == ["game_design"]\n'''
new = '''    kwargs = router.calls[0][1]\n    assert kwargs["enable_tools"] is False\n    schema = kwargs["response_schema"]\n    assert schema["required"] == ["game_design"]\n'''
if old not in source:
    raise SystemExit("game-design schema test anchor not found")
game_test.write_text(source.replace(old, new, 1), encoding="utf-8")

for rel in (
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    "minecraft_mod_ai/game_design.py",
    "tests/test_llama_structured_reasoning_stream.py",
    "tests/test_llama_cpp_adapter_request_contract.py",
    "tests/test_game_design_router.py",
):
    target = ROOT / rel
    target.write_text(target.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")

print("structured/non-tool and minimal/tool payload split applied")
