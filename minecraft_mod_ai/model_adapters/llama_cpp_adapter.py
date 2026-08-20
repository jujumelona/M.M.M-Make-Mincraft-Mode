"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. Plain text generation follows the model's own
Jinja chat template. Agent tool turns deliberately do *not* use llama.cpp's native
PEG tool parser: current Qwen-family llama.cpp builds can turn an otherwise usable
completion into HTTP 500 when the generated tool markup does not match peg-native
exactly. MMM therefore keeps tool selection/arguments host-owned over ordinary JSON
text generation and reconstructs transport-neutral ``ToolCall`` objects locally.
"""
from __future__ import annotations

import json
from dataclasses import replace
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
_HOST_TOOL_RETRY = (
    "The previous host-tool protocol response was invalid. Return exactly one valid "
    "JSON object matching the protocol below. Do not use Markdown fences, XML tool "
    "tags, prose prefixes, or prose suffixes."
)


class LlamaCppAdapter(ModelAdapter):
    """OpenAI-compatible client for the managed native llama-server."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)

    def _server_url(self, request: GenerationRequest) -> str:
        try:
            from .. import llama_server_autotune

            # The server owner must see every request. In particular, the multimodal
            # wrapper may need to replace an MMM-owned text server with an mmproj
            # server even though LLAMA_SERVER_URL already points at that text server.
            # The inner autotune path still honors healthy user-owned external URLs.
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
        if not turn.content and turn.tool_calls:
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
        """Generate one semantic turn without exposing PEG-native tool parsing.

        llama.cpp currently has real peg-native failure modes for Qwen-family agent
        turns: prefix text, malformed XML, or parser/grammar disagreement can surface
        as HTTP 500 after an expensive decode. Tool-aware MMM requests therefore use a
        host-owned JSON envelope sent as an ordinary text completion. The host validates
        the envelope and reconstructs ``ToolCall`` values. Requests without tools keep
        the normal native text path, including one bounded reasoning continuation.
        """

        cfg = self.config
        server_url = self._server_url(request)
        try:
            if request.tools:
                return _host_tool_completion(self, server_url, request)
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


def _plain_semantic_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    from ..llama_server_hardware_policy import _server_payload
    from ..llama_stream_efficiency_contract import _report_server_connection

    message = _completion_message(server_url, _server_payload(adapter, request))
    _report_server_connection(server_url)
    turn = _generation_response(message)
    if _has_semantic_action(turn):
        return turn
    if not turn.reasoning_content:
        raise RuntimeError(
            "native llama-server returned neither visible content, reasoning, nor tool calls"
        )

    continuation_request = _reasoning_continuation_request(
        request,
        turn.reasoning_content,
    )
    continued_message = _completion_message(
        server_url,
        _server_payload(adapter, continuation_request),
    )
    continued = _generation_response(continued_message)
    if not _has_semantic_action(continued):
        if continued.reasoning_content:
            raise RuntimeError(
                "native llama-server returned a reasoning-only continuation "
                "without a semantic action"
            )
        raise RuntimeError(
            "native llama-server returned no semantic action after a "
            "reasoning-only continuation"
        )
    return GenerationResponse(
        content=continued.content,
        tool_calls=continued.tool_calls,
        reasoning_content=_merge_reasoning(
            turn.reasoning_content,
            continued.reasoning_content,
        ),
    )


def _host_tool_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    """Execute one tool-aware turn through grammar-free host JSON transport."""

    from ..llama_server_hardware_policy import _server_payload
    from ..llama_stream_efficiency_contract import _report_server_connection

    last_error = ""
    for attempt in range(2):
        bridged = _host_tool_request(
            request,
            retry=attempt > 0,
            retry_error=last_error,
        )
        message = _completion_message(server_url, _server_payload(adapter, bridged))
        _report_server_connection(server_url)
        turn = _generation_response(message)
        try:
            if not turn.content:
                if turn.reasoning_content:
                    raise RuntimeError(
                        "host-tool completion returned reasoning without protocol JSON"
                    )
                raise RuntimeError("host-tool completion returned empty protocol JSON")
            return _decode_host_tool_turn(request, turn.content, turn.reasoning_content)
        except RuntimeError as exc:
            last_error = str(exc)

    raise RuntimeError(
        "native llama-server did not produce a valid host-owned tool envelope after "
        f"one bounded protocol repair: {last_error or 'unknown protocol error'}"
    )


def _host_tool_request(
    request: GenerationRequest,
    *,
    retry: bool,
    retry_error: str,
) -> GenerationRequest:
    schemas = tuple(_normalized_tool_schema(schema) for schema in request.tools)
    if not schemas:
        raise RuntimeError("host-tool bridge received no usable tool schemas")

    exact_name = _exact_forced_tool_name(request, schemas)
    if exact_name:
        selected = tuple(
            schema for schema in schemas if _schema_name(schema) == exact_name
        )
        if len(selected) != 1:
            raise RuntimeError(
                f"forced host tool {exact_name!r} does not resolve to exactly one schema"
            )
        protocol = _forced_tool_protocol(exact_name, selected[0])
    else:
        protocol = _automatic_tool_protocol(
            schemas,
            require_tool=_requires_some_tool(request.tool_choice),
            parallel=bool(request.parallel_tool_calls),
            response_format=str(request.response_format or "text"),
        )

    messages: list[Mapping[str, Any]] = [dict(message) for message in request.messages]
    if retry:
        bounded_error = " ".join(str(retry_error).split())[:400]
        messages.append(
            {
                "role": "system",
                "content": _HOST_TOOL_RETRY
                + (f" Validation error: {bounded_error}" if bounded_error else ""),
            }
        )
    messages.append({"role": "system", "content": protocol})

    # Crucial invariant: tools/tool_choice are removed before _server_payload. The
    # server therefore sees an ordinary JSON-text completion and never activates its
    # PEG-native tool grammar/parser. response_format remains host-owned; MMM does not
    # send a JSON grammar/schema sampler to llama.cpp.
    return replace(
        request,
        messages=tuple(messages),
        response_format="json",
        response_schema=None,
        tools=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )


def _forced_tool_protocol(name: str, schema: Mapping[str, Any]) -> str:
    parameters = schema.get("function", {}).get("parameters", {})
    return (
        "MMM host tool transport is active. Native llama.cpp tool parsing is disabled. "
        f"The host requires the function {name!r} now. Return ONLY the function's "
        "arguments as one JSON object. Do not return a wrapper object, function name, "
        "Markdown, XML, commentary, or final answer. The JSON object must satisfy this "
        "parameter schema:\n"
        + json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
    )


def _automatic_tool_protocol(
    schemas: Sequence[Mapping[str, Any]],
    *,
    require_tool: bool,
    parallel: bool,
    response_format: str,
) -> str:
    compact_tools = [
        {
            "name": _schema_name(schema),
            "description": str(schema.get("function", {}).get("description", "")),
            "parameters": schema.get("function", {}).get("parameters", {}),
        }
        for schema in schemas
    ]
    if require_tool:
        decision = (
            'Return {"kind":"tool","calls":[{"name":"...","arguments":{...}}]}. '
            "A final response is not legal on this turn."
        )
    else:
        decision = (
            'Return either {"kind":"tool","calls":[{"name":"...","arguments":{...}}]} '
            'or {"kind":"final","content":"..."}. '
        )
    parallel_rule = (
        "Multiple calls are allowed only when they are independent and safe to run in parallel."
        if parallel
        else "Return at most one tool call."
    )
    return (
        "MMM host tool transport is active. Native llama.cpp tool parsing is disabled. "
        "Choose the next semantic action using ONLY one JSON object and no Markdown, "
        "XML, prose prefix, or prose suffix. "
        + decision
        + " Every tool name must exactly match one available tool and every arguments "
        "value must be a JSON object matching that tool's parameter schema. "
        + parallel_rule
        + f" If kind=final, content must preserve the caller's requested {response_format!r} "
        "response format. Available tools:\n"
        + json.dumps(compact_tools, ensure_ascii=False, separators=(",", ":"))
    )


def _decode_host_tool_turn(
    request: GenerationRequest,
    content: str,
    reasoning: str,
) -> GenerationResponse:
    schemas = tuple(_normalized_tool_schema(schema) for schema in request.tools)
    available = {_schema_name(schema): schema for schema in schemas}
    if "" in available:
        available.pop("", None)
    if not available:
        raise RuntimeError("host-tool response has no authorized tool surface")

    payload = _decode_json_object(content)
    exact_name = _exact_forced_tool_name(request, schemas)
    if exact_name:
        if exact_name not in available:
            raise RuntimeError(f"forced tool {exact_name!r} is outside the authorized surface")
        arguments = payload.get("arguments") if set(payload) == {"arguments"} else payload
        if not isinstance(arguments, Mapping):
            raise RuntimeError(f"forced tool {exact_name!r} arguments must be a JSON object")
        raw_arguments = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return GenerationResponse(
            content="",
            tool_calls=(
                ToolCall(
                    id="call_0",
                    name=exact_name,
                    arguments=dict(arguments),
                    raw_arguments=raw_arguments,
                ),
            ),
            reasoning_content=reasoning.strip(),
        )

    kind = str(payload.get("kind", "")).strip().lower()
    if kind == "final":
        if _requires_some_tool(request.tool_choice):
            raise RuntimeError("host-required tool turn returned kind=final")
        final_content = payload.get("content")
        if not isinstance(final_content, str) or not final_content.strip():
            raise RuntimeError("host-tool final envelope requires non-empty string content")
        return GenerationResponse(
            content=final_content.strip(),
            reasoning_content=reasoning.strip(),
        )
    if kind != "tool":
        raise RuntimeError("host-tool envelope kind must be 'tool' or 'final'")

    raw_calls = payload.get("calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise RuntimeError("host-tool envelope kind=tool requires a non-empty calls list")
    if not request.parallel_tool_calls and len(raw_calls) != 1:
        raise RuntimeError("host-tool turn returned parallel calls when parallelism is disabled")

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            raise RuntimeError("host-tool call entry must be a JSON object")
        name = str(raw_call.get("name", "")).strip()
        if name not in available:
            raise RuntimeError(f"host-tool response selected unauthorized tool {name!r}")
        arguments = raw_call.get("arguments")
        if not isinstance(arguments, Mapping):
            raise RuntimeError(f"host-tool arguments for {name!r} must be a JSON object")
        raw_arguments = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        calls.append(
            ToolCall(
                id=f"call_{index}",
                name=name,
                arguments=dict(arguments),
                raw_arguments=raw_arguments,
            )
        )
    return GenerationResponse(
        content="",
        tool_calls=tuple(calls),
        reasoning_content=reasoning.strip(),
    )


def _decode_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("host-tool response is not one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("host-tool response must decode to one JSON object")
    return payload


def _normalized_tool_schema(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_schema = getattr(value, "to_schema", None)
    if callable(to_schema):
        schema = to_schema()
        if isinstance(schema, Mapping):
            return dict(schema)
    raise RuntimeError("host-tool bridge received an invalid tool schema")


def _schema_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _exact_forced_tool_name(
    request: GenerationRequest,
    schemas: Sequence[Mapping[str, Any]],
) -> str:
    choice = request.tool_choice
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if (
            str(choice.get("type", "")).strip() == "function"
            and isinstance(function, Mapping)
        ):
            return str(function.get("name", "")).strip()
    if str(choice or "").strip().lower() == "required" and len(schemas) == 1:
        return _schema_name(schemas[0])
    return ""


def _requires_some_tool(tool_choice: Any) -> bool:
    if isinstance(tool_choice, Mapping):
        return bool(_forced_choice_name(tool_choice))
    return str(tool_choice or "").strip().lower() == "required"


def _forced_choice_name(choice: Mapping[str, Any]) -> str:
    function = choice.get("function")
    if str(choice.get("type", "")).strip() != "function" or not isinstance(
        function, Mapping
    ):
        return ""
    return str(function.get("name", "")).strip()


def _generation_response(message: Mapping[str, Any]) -> GenerationResponse:
    content_value = message.get("content")
    content = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    return GenerationResponse(
        content=content.strip(),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        reasoning_content=reasoning.strip(),
    )


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
    return replace(
        request,
        messages=tuple(messages),
        media_paths=(),
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

    # Tool protocol turns are intentionally non-streaming so the complete host JSON
    # envelope arrives atomically. The shared client read timeout is an SSE idle
    # timeout; applying it here turns a healthy >300 s decode into ReadTimeout because
    # non-streaming responses emit no intermediate body bytes. Keep connect, write and
    # pool acquisition bounded while allowing the local decode to finish.
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    return _client(server_url).post(endpoint, json=payload, timeout=timeout)


def _bounded_response_body(response: Any, *, limit: int = 1600) -> str:
    """Keep server diagnostics bounded without echoing the model request."""

    try:
        body = str(response.text)
    except Exception:
        return ""
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RuntimeError("native llama-server tool_calls must be a list")

    result: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RuntimeError("native llama-server returned an invalid tool call")
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise RuntimeError("native llama-server tool call lacks function data")
        name = str(function.get("name", "")).strip()
        if not name:
            raise RuntimeError("native llama-server tool call lacks a function name")

        raw_value = function.get("arguments", "{}")
        if isinstance(raw_value, str):
            raw_arguments = raw_value.strip() or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"llama-server returned invalid arguments for tool {name!r}"
                ) from exc
        elif isinstance(raw_value, Mapping):
            arguments = dict(raw_value)
            raw_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            raise RuntimeError(f"Tool {name!r} arguments must be an object")
        if not isinstance(arguments, Mapping):
            raise RuntimeError(f"Tool {name!r} arguments must decode to an object")

        result.append(
            ToolCall(
                id=str(item.get("id", "")).strip() or f"call_{index}",
                name=name,
                arguments=dict(arguments),
                raw_arguments=raw_arguments,
            )
        )
    return tuple(result)
