from __future__ import annotations

import re
from pathlib import Path


ADAPTER = Path("minecraft_mod_ai/model_adapters/llama_cpp_adapter.py")
HARDWARE = Path("minecraft_mod_ai/llama_server_hardware_policy.py")
STREAM = Path("minecraft_mod_ai/llama_stream_efficiency_contract.py")
DECODE = Path("minecraft_mod_ai/llama_decode_speed_contract.py")
STRUCTURED_TEST = Path("tests/test_llama_structured_reasoning_stream.py")
ADAPTER_TEST = Path("tests/test_llama_cpp_adapter_request_contract.py")


def replace_function(source: str, name: str, replacement: str) -> str:
    start = source.find(f"\ndef {name}(")
    if start < 0:
        if source.startswith(f"def {name}("):
            start = 0
        else:
            raise SystemExit(f"function not found: {name}")
    search_from = start + 2 if start else 1
    next_def = source.find("\ndef ", search_from)
    next_class = source.find("\nclass ", search_from)
    endings = [value for value in (next_def, next_class) if value >= 0]
    end = min(endings) if endings else len(source)
    prefix = source[:start]
    if start and replacement and not replacement.startswith("\n"):
        replacement = "\n" + replacement
    return prefix + replacement.rstrip() + "\n" + source[end:]


def remove_function(source: str, name: str) -> str:
    start = source.find(f"\ndef {name}(")
    if start < 0:
        return source
    next_def = source.find("\ndef ", start + 2)
    next_class = source.find("\nclass ", start + 2)
    endings = [value for value in (next_def, next_class) if value >= 0]
    if not endings:
        return source[:start].rstrip() + "\n"
    return source[:start] + source[min(endings):]


def patch_adapter() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    turn_start = source.find("    def generate_turn(")
    if turn_start < 0:
        raise SystemExit("generate_turn not found")
    payload_start = source.find("            payload: dict[str, Any] = {", turn_start)
    response_start = source.find("            response = httpx.post(", turn_start)
    if payload_start < 0 or response_start < 0 or payload_start >= response_start:
        raise SystemExit("generate_turn payload block not found")
    payload = '''            # Native llama-server request compatibility has exactly one owner.\n            # Tool-capable turns and ordinary text turns must use the same wire shape.\n            from ..llama_server_hardware_policy import _server_payload\n\n            payload = _server_payload(self, request)\n\n'''
    source = source[:payload_start] + payload + source[response_start:]

    raise_start = source.find("            response.raise_for_status()", response_start)
    if raise_start < 0:
        if "llama server returned HTTP" not in source[response_start:]:
            raise SystemExit("completion HTTP status handling not found")
    else:
        old = "            response.raise_for_status()\n"
        new = '''            if response.status_code >= 400:\n                body = _bounded_response_body(response)\n                raise RuntimeError(\n                    f"llama server returned HTTP {response.status_code}"\n                    + (f": {body}" if body else "")\n                )\n'''
        source = source[:raise_start] + source[raise_start:].replace(old, new, 1)

    if "def _bounded_response_body(" not in source:
        marker = "\ndef _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:\n"
        helper = '''\ndef _bounded_response_body(response: Any, *, limit: int = 1600) -> str:\n    """Keep server diagnostics bounded without echoing the model request."""\n\n    try:\n        body = str(response.text)\n    except Exception:\n        return ""\n    compact = " ".join(body.split())\n    return compact if len(compact) <= limit else compact[:limit] + "..."\n\n\ndef _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:\n'''
        if marker not in source:
            raise SystemExit("tool-call helper anchor not found")
        source = source.replace(marker, helper, 1)

    compile(source, str(ADAPTER), "exec")
    ADAPTER.write_text(source, encoding="utf-8")


def patch_hardware() -> None:
    source = HARDWARE.read_text(encoding="utf-8")
    replacement = '''def _server_payload(adapter: Any, request: Any) -> dict[str, Any]:
    """Build the minimal OpenAI-compatible payload shared by every llama path.

    JSON structure is requested in the authoritative prompt and validated/repaired by
    the host. Optional OpenAI transport controls are deliberately omitted because
    native llama.cpp builds/chat templates do not support them uniformly.
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
    return payload
'''
    source = replace_function(source, "_server_payload", replacement)
    compile(source, str(HARDWARE), "exec")
    HARDWARE.write_text(source, encoding="utf-8")


def remove_duplicate_payload_owner(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = remove_function(source, "_install_host_validated_json_payload")
    source = re.sub(
        r"(?m)^\s*_install_host_validated_json_payload\([^\n]*\)\s*\n",
        "",
        source,
    )
    if "_install_host_validated_json_payload" in source:
        raise SystemExit(f"duplicate payload owner remains in {path}")
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")


def patch_structured_tests() -> None:
    source = STRUCTURED_TEST.read_text(encoding="utf-8")
    old_name = "test_json_request_disables_reasoning_without_server_grammar"
    if f"def {old_name}(" in source:
        replacement = '''def test_json_request_uses_host_validation_without_optional_transport_controls() -> None:
    request = SimpleNamespace(
        messages=(
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Plan it."},
        ),
        response_format="json",
    )
    payload = _server_payload(_adapter(), request)
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "parallel_tool_calls" not in payload
    assert payload["max_tokens"] == 8192


def test_tool_request_uses_same_minimal_native_payload() -> None:
    request = SimpleNamespace(
        messages=({"role": "user", "content": "inspect then plan"},),
        response_format="json",
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
'''
        source = replace_function(source, old_name, replacement)
    compile(source, str(STRUCTURED_TEST), "exec")
    STRUCTURED_TEST.write_text(source, encoding="utf-8")


def write_adapter_tests() -> None:
    content = '''from __future__ import annotations

import httpx
import pytest

from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    GenerationRequest,
    ModelBackendError,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


class _HealthResponse:
    status_code = 200

    @staticmethod
    def raise_for_status() -> None:
        return None


class _CompletionResponse:
    def __init__(self, *, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="planner",
            adapter="llama_cpp",
            model_id="test/qwen35",
            max_new_tokens=512,
        )
    )


def test_generate_turn_reuses_canonical_payload_for_json_tools(monkeypatch) -> None:
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


def test_generate_turn_preserves_llama_server_400_body_without_prompt(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _HealthResponse())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _CompletionResponse(
            status_code=400,
            text='{"error":{"message":"unsupported request field"}}',
        ),
    )

    with pytest.raises(ModelBackendError) as caught:
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "SECRET_PROMPT_SENTINEL"},),
                response_format="json",
            )
        )

    message = str(caught.value)
    assert "HTTP 400" in message
    assert "unsupported request field" in message
    assert "SECRET_PROMPT_SENTINEL" not in message
'''
    compile(content, str(ADAPTER_TEST), "exec")
    ADAPTER_TEST.write_text(content, encoding="utf-8")


def main() -> None:
    patch_adapter()
    patch_hardware()
    remove_duplicate_payload_owner(STREAM)
    remove_duplicate_payload_owner(DECODE)
    patch_structured_tests()
    write_adapter_tests()


if __name__ == "__main__":
    main()
