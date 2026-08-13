from __future__ import annotations

from pathlib import Path


ADAPTER = Path("minecraft_mod_ai/model_adapters/llama_cpp_adapter.py")
HARDWARE = Path("minecraft_mod_ai/llama_server_hardware_policy.py")
STREAM = Path("minecraft_mod_ai/llama_stream_efficiency_contract.py")
STRUCTURED_TEST = Path("tests/test_llama_structured_reasoning_stream.py")
ADAPTER_TEST = Path("tests/test_llama_cpp_adapter_request_contract.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"repair anchor not found: {label}")
    return text.replace(old, new, 1)


def patch_adapter() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    old_payload = '''            payload: dict[str, Any] = {\n                "model": "local",\n                "messages": [dict(message) for message in request.messages],\n                "max_tokens": int(cfg.max_new_tokens),\n                "temperature": 0.0,\n            }\n            if getattr(request, "response_format", None) == "json":\n                payload["response_format"] = {"type": "json_object"}\n                payload["reasoning_effort"] = "none"\n            if request.tools:\n                payload["tools"] = [dict(tool) for tool in request.tools]\n                payload["tool_choice"] = request.tool_choice or "auto"\n                payload["parallel_tool_calls"] = bool(request.parallel_tool_calls)\n\n'''
    new_payload = '''            # One owner builds every native llama-server request.  Tool-capable\n            # planner turns must not bypass the same compatibility contract used by\n            # ordinary streamed text generation.\n            from ..llama_server_hardware_policy import _server_payload\n\n            payload = _server_payload(self, request)\n\n'''
    source = replace_once(source, old_payload, new_payload, "adapter shared payload")

    old_post = '''            response = httpx.post(\n                f"{server_url}/chat/completions",\n                json=payload,\n                timeout=None,\n            )\n            response.raise_for_status()\n            data = response.json()\n'''
    new_post = '''            response = httpx.post(\n                f"{server_url}/chat/completions",\n                json=payload,\n                timeout=None,\n            )\n            if response.status_code >= 400:\n                body = _bounded_response_body(response)\n                raise RuntimeError(\n                    f"llama server returned HTTP {response.status_code}"\n                    + (f": {body}" if body else "")\n                )\n            data = response.json()\n'''
    source = replace_once(source, old_post, new_post, "adapter HTTP diagnostics")

    helper_anchor = '''\ndef _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:\n'''
    helper = '''\ndef _bounded_response_body(response: Any, *, limit: int = 1600) -> str:\n    """Return a bounded one-line server diagnostic without echoing the request."""\n\n    try:\n        body = str(response.text)\n    except Exception:\n        return ""\n    compact = " ".join(body.split())\n    if len(compact) > limit:\n        return compact[:limit] + "..."\n    return compact\n\n\ndef _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:\n'''
    source = replace_once(source, helper_anchor, helper, "adapter bounded error helper")
    compile(source, str(ADAPTER), "exec")
    ADAPTER.write_text(source, encoding="utf-8")


def patch_hardware_payload() -> None:
    source = HARDWARE.read_text(encoding="utf-8")
    old = '''def _server_payload(adapter: Any, request: Any) -> dict[str, Any]:\n    payload: dict[str, Any] = {\n        "model": "local",\n        "messages": [dict(message) for message in request.messages],\n        "max_tokens": int(adapter.config.max_new_tokens),\n        "temperature": 0.0,\n    }\n    if getattr(request, "response_format", None) == "json":\n        payload["response_format"] = {"type": "json_object"}\n        payload["reasoning_effort"] = "none"\n    return payload\n'''
    new = '''def _server_payload(adapter: Any, request: Any) -> dict[str, Any]:\n    """Build the common-subset OpenAI chat payload accepted by managed llama-server.\n\n    JSON is requested in the authoritative system prompt and validated by the host.\n    Do not send optional OpenAI transport controls such as ``response_format``,\n    ``reasoning_effort`` or ``parallel_tool_calls`` here: native llama.cpp builds and\n    model chat templates differ in support for those fields, and tool-capable planner\n    turns must use the exact same wire contract as ordinary local generation.\n    """\n\n    payload: dict[str, Any] = {\n        "model": "local",\n        "messages": [dict(message) for message in request.messages],\n        "max_tokens": int(adapter.config.max_new_tokens),\n        "temperature": 0.0,\n    }\n    tools = getattr(request, "tools", ()) or ()\n    if tools:\n        payload["tools"] = [dict(tool) for tool in tools]\n        tool_choice = getattr(request, "tool_choice", None)\n        if tool_choice is not None:\n            payload["tool_choice"] = (\n                dict(tool_choice) if isinstance(tool_choice, dict) else tool_choice\n            )\n    return payload\n'''
    source = replace_once(source, old, new, "canonical llama server payload")
    compile(source, str(HARDWARE), "exec")
    HARDWARE.write_text(source, encoding="utf-8")


def patch_stream_contract() -> None:
    source = STREAM.read_text(encoding="utf-8")
    old = '''def _install_host_validated_json_payload(hardware_module: Any) -> None:\n    current_payload = hardware_module._server_payload\n    if getattr(current_payload, "_mmm_host_validated_json_no_gbnf", False):\n        return\n\n    @wraps(current_payload)\n    def host_validated_payload(adapter: Any, request: Any) -> dict[str, Any]:\n        payload = dict(current_payload(adapter, request))\n        if getattr(request, "response_format", None) == "json":\n            payload.pop("response_format", None)\n            payload["reasoning_effort"] = "none"\n        return payload\n\n    host_validated_payload._mmm_host_validated_json_no_gbnf = True\n    hardware_module._server_payload = host_validated_payload\n\n\n'''
    source = replace_once(source, old, "", "remove duplicate payload wrapper")
    source = replace_once(
        source,
        '''    _install_host_validated_json_payload(hardware_module)\n\n''',
        "",
        "remove duplicate payload install",
    )
    compile(source, str(STREAM), "exec")
    STREAM.write_text(source, encoding="utf-8")


def patch_structured_tests() -> None:
    source = STRUCTURED_TEST.read_text(encoding="utf-8")
    old = '''def test_json_request_disables_reasoning_without_server_grammar() -> None:\n    request = SimpleNamespace(\n        messages=(\n            {"role": "system", "content": "Return JSON."},\n            {"role": "user", "content": "Plan it."},\n        ),\n        response_format="json",\n    )\n    payload = _server_payload(_adapter(), request)\n    assert "response_format" not in payload\n    assert payload["reasoning_effort"] == "none"\n    assert payload["max_tokens"] == 8192\n\n\n'''
    new = '''def test_json_request_uses_host_validation_without_optional_transport_controls() -> None:\n    request = SimpleNamespace(\n        messages=(\n            {"role": "system", "content": "Return JSON."},\n            {"role": "user", "content": "Plan it."},\n        ),\n        response_format="json",\n    )\n    payload = _server_payload(_adapter(), request)\n    assert "response_format" not in payload\n    assert "reasoning_effort" not in payload\n    assert "parallel_tool_calls" not in payload\n    assert payload["max_tokens"] == 8192\n\n\ndef test_tool_request_uses_the_same_minimal_native_payload() -> None:\n    request = SimpleNamespace(\n        messages=({"role": "user", "content": "inspect then plan"},),\n        response_format="json",\n        tools=(\n            {\n                "type": "function",\n                "function": {\n                    "name": "lookup",\n                    "description": "lookup evidence",\n                    "parameters": {"type": "object", "properties": {}},\n                },\n            },\n        ),\n        tool_choice="auto",\n        parallel_tool_calls=True,\n    )\n    payload = _server_payload(_adapter(), request)\n    assert payload["tools"][0]["function"]["name"] == "lookup"\n    assert payload["tool_choice"] == "auto"\n    assert "response_format" not in payload\n    assert "reasoning_effort" not in payload\n    assert "parallel_tool_calls" not in payload\n\n\n'''
    source = replace_once(source, old, new, "structured payload tests")
    compile(source, str(STRUCTURED_TEST), "exec")
    STRUCTURED_TEST.write_text(source, encoding="utf-8")


def write_adapter_tests() -> None:
    content = '''from __future__ import annotations\n\nimport httpx\nimport pytest\n\nfrom minecraft_mod_ai.model_adapters.base import (\n    AdapterConfig,\n    GenerationRequest,\n    ModelBackendError,\n)\nfrom minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter\n\n\nclass _HealthResponse:\n    status_code = 200\n\n    @staticmethod\n    def raise_for_status() -> None:\n        return None\n\n\nclass _CompletionResponse:\n    def __init__(self, *, status_code: int, payload=None, text: str = "") -> None:\n        self.status_code = status_code\n        self._payload = payload\n        self.text = text\n\n    def json(self):\n        return self._payload\n\n\ndef _adapter() -> LlamaCppAdapter:\n    return LlamaCppAdapter(\n        AdapterConfig(\n            role="planner",\n            adapter="llama_cpp",\n            model_id="test/qwen35",\n            max_new_tokens=512,\n        )\n    )\n\n\ndef test_generate_turn_reuses_canonical_payload_for_json_tools(monkeypatch) -> None:\n    captured: dict[str, object] = {}\n    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")\n    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())\n\n    def post(url, *, json, timeout):\n        captured["url"] = url\n        captured["payload"] = json\n        return _CompletionResponse(\n            status_code=200,\n            payload={\n                "choices": [\n                    {"message": {"content": '{"game_design":{}}'}}\n                ]\n            },\n        )\n\n    monkeypatch.setattr(httpx, "post", post)\n    request = GenerationRequest(\n        messages=({"role": "user", "content": "plan"},),\n        response_format="json",\n        tools=(\n            {\n                "type": "function",\n                "function": {\n                    "name": "lookup",\n                    "description": "lookup",\n                    "parameters": {"type": "object", "properties": {}},\n                },\n            },\n        ),\n        tool_choice="auto",\n        parallel_tool_calls=True,\n    )\n\n    turn = _adapter().generate_turn(request)\n\n    assert turn.content == '{"game_design":{}}'\n    payload = captured["payload"]\n    assert payload["tools"][0]["function"]["name"] == "lookup"\n    assert payload["tool_choice"] == "auto"\n    assert "response_format" not in payload\n    assert "reasoning_effort" not in payload\n    assert "parallel_tool_calls" not in payload\n\n\ndef test_generate_turn_preserves_llama_server_400_body(monkeypatch) -> None:\n    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")\n    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())\n    monkeypatch.setattr(\n        httpx,\n        "post",\n        lambda *args, **kwargs: _CompletionResponse(\n            status_code=400,\n            text='{"error":{"message":"unsupported request field"}}',\n        ),\n    )\n\n    with pytest.raises(ModelBackendError) as caught:\n        _adapter().generate_turn(\n            GenerationRequest(\n                messages=({"role": "user", "content": "plan"},),\n                response_format="json",\n            )\n        )\n\n    message = str(caught.value)\n    assert "HTTP 400" in message\n    assert "unsupported request field" in message\n    assert "plan" not in message\n'''
    ADAPTER_TEST.write_text(content, encoding="utf-8")
    compile(content, str(ADAPTER_TEST), "exec")


def main() -> None:
    patch_adapter()
    patch_hardware_payload()
    patch_stream_contract()
    patch_structured_tests()
    write_adapter_tests()


if __name__ == "__main__":
    main()
