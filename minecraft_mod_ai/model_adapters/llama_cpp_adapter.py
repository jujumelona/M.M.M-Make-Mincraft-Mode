"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. Tool-aware turns keep the model's own Jinja
``tools`` prompt, but deliberately disable llama.cpp's PEG-native tool parser and
parse the model's native Qwen tagged tool calls on the host. This keeps one decode,
the same sampling/MTP path, and one canonical model-facing tool protocol.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import httpx

from .base import (
    AdapterConfig,
    GenerationRequest,
    GenerationResponse,
    ModelAdapter,
    ModelBackendError,
    ToolCall,
)


_DEFAULT_HTTPX_POST = httpx.post
_REASONING_CONTINUATION = (
    "Continue from the reasoning above and complete this same assistant turn now. "
    "Call an available tool if evidence or an action is required; otherwise return "
    "the requested final answer. Do not return another reasoning-only response."
)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_FUNCTION_OPEN = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_STRUCTURAL_MARKERS = (
    _TOOL_CALL_OPEN,
    _TOOL_CALL_CLOSE,
    _FUNCTION_OPEN,
    _FUNCTION_CLOSE,
    _PARAMETER_OPEN,
    _PARAMETER_CLOSE,
)


class LlamaCppAdapter(ModelAdapter):
    """OpenAI-compatible client for the managed native llama-server."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        _bind_peg_free_runtime_guard()

    def _server_url(self, request: GenerationRequest) -> str:
        try:
            from .. import llama_server_autotune

            selected = llama_server_autotune.ensure_tuned_server(self.config, request)
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=RuntimeError(
                    "native llama-server could not be prepared; local GGUF inference "
                    "has no alternate in-process backend"
                ),
            ) from exc
        endpoint = (selected or "").strip().rstrip("/")
        if not endpoint:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=RuntimeError(
                    "native llama-server is required for local GGUF inference but no "
                    "server URL was produced"
                ),
            )
        return endpoint

    def generate(self, request: GenerationRequest) -> str:
        turn = self.generate_turn(request)
        if turn.tool_calls:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=(
                    "A tool-aware completion was requested through the text-only "
                    "generate() API. Use ModelRouter.generate_text() so tool calls "
                    "can be executed."
                ),
            )
        return turn.content

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one semantic turn without llama.cpp PEG-native tool parsing."""

        cfg = self.config
        server_url = self._server_url(request)
        try:
            if request.tools:
                return _tool_semantic_completion(self, server_url, request)
            return _plain_semantic_completion(self, server_url, request)
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc

    def close(self) -> None:
        return None


def _bind_peg_free_runtime_guard() -> None:
    """Seal legacy/direct server entry points so no tool request can re-enable PEG.

    Hardware policy already owns runtime adapter binding. Keep that architecture, but
    harden its two generic hooks once: every tool-bearing payload gets transport
    ``tool_choice=none``, and its text-only streaming shortcut delegates tool turns to
    ``generate_turn`` instead of streaming raw Qwen markup.
    """

    from .. import llama_server_hardware_policy as policy

    current_payload = policy._server_payload
    if not getattr(current_payload, "_mmm_peg_free_tools", False):
        original_payload = current_payload

        def peg_free_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
            payload = original_payload(adapter, request)
            if getattr(request, "tools", ()):
                payload["tool_choice"] = "none"
            return payload

        peg_free_server_payload._mmm_peg_free_tools = True  # type: ignore[attr-defined]
        policy._server_payload = peg_free_server_payload

    current_strict = policy._strict_server_generate
    if not getattr(current_strict, "_mmm_peg_free_tool_guard", False):
        original_strict = current_strict

        def peg_free_strict_generate(
            adapter: Any,
            request: Any,
            server_url: str,
        ) -> str:
            if getattr(request, "tools", ()):
                turn = adapter.generate_turn(request)
                if turn.tool_calls:
                    raise ModelBackendError(
                        role=adapter.config.role,
                        model_id=adapter.config.model_id,
                        cause=(
                            "A tool-aware completion reached the text-only streaming "
                            "API. Use ModelRouter.generate_text() so calls are executed."
                        ),
                    )
                return turn.content
            return original_strict(adapter, request, server_url)

        peg_free_strict_generate._mmm_peg_free_tool_guard = True  # type: ignore[attr-defined]
        policy._strict_server_generate = peg_free_strict_generate


def _plain_semantic_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    from ..llama_server_hardware_policy import _server_payload
    from ..llama_stream_efficiency_contract import _report_server_connection

    message = _completion_message(server_url, _server_payload(adapter, request))
    _report_server_connection(server_url)
    turn = _plain_generation_response(message)
    if _has_semantic_action(turn):
        return turn
    if not turn.reasoning_content:
        raise RuntimeError(
            "native llama-server returned neither visible content nor reasoning"
        )

    continuation_request = _reasoning_continuation_request(
        request,
        turn.reasoning_content,
    )
    continued_message = _completion_message(
        server_url,
        _server_payload(adapter, continuation_request),
    )
    continued = _plain_generation_response(continued_message)
    if not _has_semantic_action(continued):
        if continued.reasoning_content:
            raise RuntimeError(
                "native llama-server returned a reasoning-only continuation without "
                "a semantic action"
            )
        raise RuntimeError(
            "native llama-server returned no semantic action after a reasoning-only "
            "continuation"
        )
    return GenerationResponse(
        content=continued.content,
        reasoning_content=_merge_reasoning(
            turn.reasoning_content,
            continued.reasoning_content,
        ),
    )


def _tool_semantic_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    """Run one normal Qwen tool turn while keeping llama.cpp PEG fully inactive."""

    from ..llama_stream_efficiency_contract import _report_server_connection

    message = _completion_message(server_url, _peg_free_tool_payload(adapter, request))
    _report_server_connection(server_url)
    turn = _qwen_tool_generation_response(message, request)
    if _has_semantic_action(turn):
        return turn
    if not turn.reasoning_content:
        raise RuntimeError(
            "native llama-server returned neither visible content, reasoning, nor "
            "Qwen tool calls"
        )

    continuation_request = _reasoning_continuation_request(
        request,
        turn.reasoning_content,
    )
    continued_message = _completion_message(
        server_url,
        _peg_free_tool_payload(adapter, continuation_request),
    )
    continued = _qwen_tool_generation_response(
        continued_message,
        continuation_request,
    )
    if not _has_semantic_action(continued):
        if continued.reasoning_content:
            raise RuntimeError(
                "native llama-server returned a reasoning-only tool continuation "
                "without a semantic action"
            )
        raise RuntimeError(
            "native llama-server returned no semantic action after a reasoning-only "
            "tool continuation"
        )
    return GenerationResponse(
        content=continued.content,
        tool_calls=continued.tool_calls,
        reasoning_content=_merge_reasoning(
            turn.reasoning_content,
            continued.reasoning_content,
        ),
    )


def _peg_free_tool_payload(
    adapter: LlamaCppAdapter,
    request: GenerationRequest,
) -> dict[str, Any]:
    """Keep native Jinja/tools but disable llama.cpp's PEG-native tool machinery."""

    from ..llama_server_hardware_policy import _server_payload

    payload = _server_payload(adapter, request)
    if not payload.get("tools"):
        raise RuntimeError("PEG-free tool transport received no tool schemas")
    payload["tool_choice"] = "none"
    return payload


def _plain_generation_response(message: Mapping[str, Any]) -> GenerationResponse:
    if message.get("tool_calls"):
        raise RuntimeError("plain completion unexpectedly returned tool_calls")
    content_value = message.get("content")
    content = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    return GenerationResponse(
        content=content.strip(),
        reasoning_content=reasoning.strip(),
    )


def _qwen_tool_generation_response(
    message: Mapping[str, Any],
    request: GenerationRequest,
) -> GenerationResponse:
    if message.get("tool_calls"):
        raise RuntimeError(
            "llama-server returned server-parsed tool_calls even though PEG-free "
            "transport disabled server tool parsing"
        )

    schemas = _tool_schema_map(request.tools)
    content_value = message.get("content")
    content_raw = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    reasoning_raw = reasoning_value if isinstance(reasoning_value, str) else ""

    reasoning, reasoning_calls = _parse_qwen_tool_markup(reasoning_raw, schemas)
    content, content_calls = _parse_qwen_tool_markup(content_raw, schemas)
    calls = (*reasoning_calls, *content_calls)
    _validate_tool_choice(request, calls)

    return GenerationResponse(
        content=content.strip(),
        tool_calls=tuple(calls),
        reasoning_content=reasoning.strip(),
    )


def _tool_schema_map(
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for schema in schemas:
        function = schema.get("function")
        if not isinstance(function, Mapping):
            raise RuntimeError("tool schema lacks function metadata")
        name = str(function.get("name", "")).strip()
        if not name:
            raise RuntimeError("tool schema lacks a function name")
        if name in result:
            raise RuntimeError(f"duplicate tool schema name {name!r}")
        parameters = function.get("parameters", {})
        if parameters is not None and not isinstance(parameters, Mapping):
            raise RuntimeError(f"tool {name!r} parameters schema must be an object")
        result[name] = dict(parameters or {})
    return result


def _parse_qwen_tool_markup(
    text: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[str, tuple[ToolCall, ...]]:
    """Parse Qwen3.5 native tagged tool calls without regex or PEG.

    Both the official ``<tool_call><function=...>`` form and the wrapper-omitted
    ``<function=...>`` form are accepted. Parameter terminators are recognized only
    at structural boundaries so Java/source strings containing XML-looking text are
    not truncated merely because they contain ``</parameter>``.
    """

    if not text:
        return "", ()

    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        wrapped_at = text.find(_TOOL_CALL_OPEN, cursor)
        direct_at = text.find(_FUNCTION_OPEN, cursor)
        starts = [value for value in (wrapped_at, direct_at) if value >= 0]
        if not starts:
            break
        start = min(starts)
        wrapped = wrapped_at == start
        function_at = start + len(_TOOL_CALL_OPEN) if wrapped else start
        function_at = _skip_space(text, function_at)
        if not text.startswith(_FUNCTION_OPEN, function_at):
            if wrapped:
                raise RuntimeError("Qwen tool_call block does not begin with a function")
            cursor = start + 1
            continue

        call, end = _parse_qwen_function(
            text,
            function_at,
            schemas,
            call_index=len(calls),
        )
        if wrapped:
            close_at = _skip_space(text, end)
            if not text.startswith(_TOOL_CALL_CLOSE, close_at):
                raise RuntimeError("Qwen tool_call block is missing </tool_call>")
            end = close_at + len(_TOOL_CALL_CLOSE)
        calls.append(call)
        spans.append((start, end))
        cursor = end

    for marker in _STRUCTURAL_MARKERS:
        pos = text.find(marker)
        if pos >= 0 and not any(begin <= pos < end for begin, end in spans):
            raise RuntimeError(f"unparsed Qwen tool markup begins at {marker!r}")

    if not spans:
        return text, ()
    visible: list[str] = []
    previous = 0
    for begin, end in spans:
        visible.append(text[previous:begin])
        previous = end
    visible.append(text[previous:])
    return "".join(visible), tuple(calls)


def _parse_qwen_function(
    text: str,
    start: int,
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    call_index: int,
) -> tuple[ToolCall, int]:
    name_start = start + len(_FUNCTION_OPEN)
    name_end = text.find(">", name_start)
    if name_end < 0:
        raise RuntimeError("Qwen function tag is missing '>'")
    name = text[name_start:name_end].strip()
    if not name:
        raise RuntimeError("Qwen function tag has an empty tool name")
    schema = schemas.get(name)
    if schema is None:
        raise RuntimeError(f"Qwen requested an unexposed tool {name!r}")

    properties_value = schema.get("properties", {})
    properties = properties_value if isinstance(properties_value, Mapping) else {}
    required_value = schema.get("required", ())
    required: set[str] = set()
    if isinstance(required_value, Sequence) and not isinstance(
        required_value, (str, bytes)
    ):
        required = {str(value) for value in required_value}
    additional = schema.get("additionalProperties", True)

    arguments: dict[str, Any] = {}
    pos = name_end + 1
    while True:
        pos = _skip_space(text, pos)
        if text.startswith(_FUNCTION_CLOSE, pos):
            end = pos + len(_FUNCTION_CLOSE)
            break
        if not text.startswith(_PARAMETER_OPEN, pos):
            snippet = " ".join(text[pos : pos + 120].split())
            raise RuntimeError(
                f"Qwen tool {name!r} emitted invalid parameter structure near {snippet!r}"
            )

        key_start = pos + len(_PARAMETER_OPEN)
        key_end = text.find(">", key_start)
        if key_end < 0:
            raise RuntimeError(f"Qwen tool {name!r} parameter tag is missing '>'")
        key = text[key_start:key_end].strip()
        if not key:
            raise RuntimeError(f"Qwen tool {name!r} emitted an empty parameter name")
        if key in arguments:
            raise RuntimeError(f"Qwen tool {name!r} repeated parameter {key!r}")
        if key not in properties and additional is False:
            raise RuntimeError(f"Qwen tool {name!r} emitted unknown parameter {key!r}")

        value_start = key_end + 1
        close_at = _find_parameter_close(text, value_start)
        if close_at < 0:
            raise RuntimeError(
                f"Qwen tool {name!r} parameter {key!r} is missing a structural "
                "</parameter> terminator"
            )
        raw = _unwrap_parameter_text(text[value_start:close_at])
        value_schema = properties.get(key, {})
        if not isinstance(value_schema, Mapping):
            value_schema = {}
        arguments[key] = _decode_parameter_value(name, key, raw, value_schema)
        pos = close_at + len(_PARAMETER_CLOSE)

    missing = sorted(required - arguments.keys())
    if missing:
        raise RuntimeError(
            f"Qwen tool {name!r} omitted required parameters: {', '.join(missing)}"
        )

    raw_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        f"{call_index}\0{name}\0{raw_arguments}".encode("utf-8")
    ).hexdigest()[:16]
    return (
        ToolCall(
            id=f"call_{digest}",
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        ),
        end,
    )


def _find_parameter_close(text: str, start: int) -> int:
    search = start
    while True:
        candidate = text.find(_PARAMETER_CLOSE, search)
        if candidate < 0:
            return -1
        after = _skip_space(text, candidate + len(_PARAMETER_CLOSE))
        if (
            text.startswith(_PARAMETER_OPEN, after)
            or text.startswith(_FUNCTION_CLOSE, after)
            or text.startswith(_TOOL_CALL_CLOSE, after)
        ):
            return candidate
        search = candidate + len(_PARAMETER_CLOSE)


def _unwrap_parameter_text(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def _decode_parameter_value(
    tool_name: str,
    key: str,
    raw: str,
    schema: Mapping[str, Any],
) -> Any:
    expected = _schema_value_type(schema)
    compact = raw.strip()
    try:
        if expected == "string":
            value: Any = raw
        elif expected == "integer":
            if not compact or any(ch in compact.lower() for ch in (".", "e")):
                raise ValueError("not an integer")
            value = int(compact)
        elif expected == "number":
            value = float(compact)
        elif expected == "boolean":
            lowered = compact.lower()
            if lowered not in {"true", "false"}:
                raise ValueError("not a boolean")
            value = lowered == "true"
        elif expected == "null":
            if compact.lower() != "null":
                raise ValueError("not null")
            value = None
        elif expected in {"object", "array"}:
            value = json.loads(compact)
            if expected == "object" and not isinstance(value, Mapping):
                raise ValueError("not an object")
            if expected == "array" and not isinstance(value, list):
                raise ValueError("not an array")
        else:
            if compact.startswith(("{", "[", '"')) or compact in {
                "true",
                "false",
                "null",
            }:
                value = json.loads(compact)
            else:
                value = raw
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Qwen tool {tool_name!r} emitted invalid {expected or 'schema'} value "
            f"for parameter {key!r}"
        ) from exc

    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        raise RuntimeError(
            f"Qwen tool {tool_name!r} emitted value outside enum for parameter {key!r}"
        )
    return value


def _schema_value_type(schema: Mapping[str, Any]) -> str:
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return raw_type
    if isinstance(raw_type, list):
        non_null = [str(value) for value in raw_type if str(value) != "null"]
        if len(non_null) == 1:
            return non_null[0]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        kinds = {_json_type(value) for value in enum if value is not None}
        if len(kinds) == 1:
            return next(iter(kinds))
    for keyword in ("oneOf", "anyOf"):
        choices = schema.get(keyword)
        if isinstance(choices, list):
            kinds = {
                _schema_value_type(choice)
                for choice in choices
                if isinstance(choice, Mapping)
            }
            kinds.discard("")
            kinds.discard("null")
            if len(kinds) == 1:
                return next(iter(kinds))
    return ""


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return ""


def _validate_tool_choice(
    request: GenerationRequest,
    calls: Sequence[ToolCall],
) -> None:
    if not request.parallel_tool_calls and len(calls) > 1:
        raise RuntimeError("model emitted parallel tool calls when they are disabled")

    choice = request.tool_choice
    if choice is None or choice == "auto":
        return
    if choice == "none":
        if calls:
            raise RuntimeError("model emitted a tool call when tool_choice is none")
        return
    if choice == "required":
        if not calls:
            raise RuntimeError("model did not emit a tool call when one is required")
        return
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if not isinstance(function, Mapping):
            raise RuntimeError("named tool_choice lacks function metadata")
        expected = str(function.get("name", "")).strip()
        if not expected:
            raise RuntimeError("named tool_choice lacks a function name")
        if len(calls) != 1 or calls[0].name != expected:
            received = ", ".join(call.name for call in calls) or "<none>"
            raise RuntimeError(
                f"model violated named tool_choice {expected!r}; received {received}"
            )
        return
    raise RuntimeError(f"unsupported tool_choice contract: {choice!r}")


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _has_semantic_action(turn: GenerationResponse) -> bool:
    return bool(turn.content or turn.tool_calls)


def _reasoning_continuation_request(
    request: GenerationRequest,
    reasoning: str,
) -> GenerationRequest:
    messages = [dict(message) for message in request.messages]
    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": reasoning,
            },
            {"role": "user", "content": _REASONING_CONTINUATION},
        ]
    )
    return GenerationRequest(
        messages=tuple(messages),
        media_paths=(),
        response_format=request.response_format,
        response_schema=request.response_schema,
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
    )


def _merge_reasoning(first: str, second: str) -> str:
    first = first.strip()
    second = second.strip()
    if not first:
        return second
    if not second or second == first:
        return first
    return f"{first}\n{second}"


def _completion_message(server_url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    response = _post_completion(server_url, payload)
    if response.status_code >= 400:
        body = _bounded_response_body(response)
        raise RuntimeError(
            f"llama server returned HTTP {response.status_code}"
            + (f": {body}" if body else "")
        )
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("native llama-server returned no completion choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("native llama-server returned an invalid completion choice")
    finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        raise RuntimeError(
            "native llama-server reached its model/server context boundary before "
            "the assistant turn completed"
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("native llama-server returned no assistant message")
    return message


def _post_completion(server_url: str, payload: Mapping[str, Any]) -> Any:
    """Use the persistent pool without timing out a healthy long local decode."""

    endpoint = f"{server_url}/chat/completions"
    if httpx.post is not _DEFAULT_HTTPX_POST:
        return httpx.post(endpoint, json=payload, timeout=None)
    from ..llama_stream_efficiency_contract import _client

    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    return _client(server_url).post(endpoint, json=payload, timeout=timeout)


def _bounded_response_body(response: Any, *, limit: int = 1600) -> str:
    try:
        body = str(response.text)
    except Exception:
        return ""
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."
