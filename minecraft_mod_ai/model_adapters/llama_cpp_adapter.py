"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. The Python llama binding is intentionally not
an execution fallback: model selection, GPU residency, MTP benchmarking and metrics
all belong to the managed native llama-server process.
"""
from __future__ import annotations

import json
import os
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
_HOST_TOOL_PROTOCOL_VERSION = "mmm/host-tool-envelope-v1"
_MAX_HOST_TOOL_REPAIR_ATTEMPTS = 1


class LlamaCppAdapter(ModelAdapter):
    """OpenAI-compatible client for the managed native llama-server.

    llama.cpp native tool/JSON controls are intentionally not used. Some server/chat
    template combinations compile those controls into GBNF and can fail before the
    model runs. Tool selection remains model-driven, but the wire contract is an
    ordinary JSON envelope validated and executed by the host.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)

    def _server_url(self, request: GenerationRequest) -> str:
        explicit = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
        if explicit:
            return explicit
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
        """Generate one assistant turn without native llama.cpp grammar controls."""

        cfg = self.config
        server_url = self._server_url(request)
        try:
            from ..llama_server_hardware_policy import _server_payload
            from ..llama_stream_efficiency_contract import _report_server_connection

            if request.tools:
                turn = self._generate_host_tool_turn(
                    server_url,
                    request,
                    payload_builder=_server_payload,
                )
                _report_server_connection(server_url)
                return turn

            payload = _server_payload(self, request)
            message = _completion_message(server_url, payload)
            _report_server_connection(server_url)
            content_value = message.get("content")
            content = content_value if isinstance(content_value, str) else ""
            reasoning_value = message.get("reasoning_content")
            reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
            tool_calls = _parse_tool_calls(message.get("tool_calls"))
            if not content.strip() and not tool_calls:
                raise RuntimeError(
                    "native llama-server returned neither visible content nor tool calls"
                )
            return GenerationResponse(
                content=content.strip(),
                tool_calls=tool_calls,
                reasoning_content=reasoning.strip(),
            )
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc

    def _generate_host_tool_turn(
        self,
        server_url: str,
        request: GenerationRequest,
        *,
        payload_builder: Any,
    ) -> GenerationResponse:
        allowed = frozenset(_tool_schema_names(request.tools))
        if not allowed:
            raise RuntimeError("Tool-capable llama request contains no callable tool names")

        wire_messages = _host_tool_messages(request.messages, request.tools)
        wire_request = GenerationRequest(
            messages=wire_messages,
            media_paths=request.media_paths,
            response_format="json",
            response_schema=None,
            tools=(),
            tool_choice=None,
            parallel_tool_calls=False,
        )
        payload = payload_builder(self, wire_request)
        _assert_no_native_grammar_controls(payload)
        message = _completion_message(server_url, payload)
        raw = str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning_content") or "").strip()

        error: Exception | None = None
        for repair_attempt in range(_MAX_HOST_TOOL_REPAIR_ATTEMPTS + 1):
            try:
                return _parse_host_tool_envelope(raw, allowed, reasoning=reasoning)
            except (json.JSONDecodeError, RuntimeError, ValueError, TypeError) as exc:
                error = exc
                if repair_attempt >= _MAX_HOST_TOOL_REPAIR_ATTEMPTS:
                    break
                repair_messages = [
                    *wire_messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "system",
                        "content": (
                            "The previous host tool envelope was invalid. Repair only its "
                            "syntax/shape. Return exactly one JSON object, no markdown and no "
                            "explanation. Preserve the intended tool names and arguments; use "
                            "kind='final' only if the previous output was already a final answer."
                        ),
                    },
                ]
                repair_request = GenerationRequest(
                    messages=repair_messages,
                    response_format="json",
                    tools=(),
                    tool_choice=None,
                    parallel_tool_calls=False,
                )
                repair_payload = payload_builder(self, repair_request)
                _assert_no_native_grammar_controls(repair_payload)
                repair_message = _completion_message(server_url, repair_payload)
                raw = str(repair_message.get("content") or "").strip()
                repair_reasoning = repair_message.get("reasoning_content")
                if isinstance(repair_reasoning, str) and repair_reasoning.strip():
                    reasoning = repair_reasoning.strip()

        raise RuntimeError(
            "Host-owned llama tool envelope remained invalid after bounded repair: "
            f"{type(error).__name__}: {error}"
        )

    def close(self) -> None:
        return None


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
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, Mapping):
        raise RuntimeError("native llama-server returned no assistant message")
    return message


def _post_completion(server_url: str, payload: Mapping[str, Any]) -> Any:
    """Use the persistent production pool while preserving injected HTTP transports."""

    endpoint = f"{server_url}/chat/completions"
    if httpx.post is not _DEFAULT_HTTPX_POST:
        return httpx.post(endpoint, json=payload, timeout=None)
    from ..llama_stream_efficiency_contract import _client

    return _client(server_url).post(endpoint, json=payload)


def _bounded_response_body(response: Any, *, limit: int = 1600) -> str:
    """Keep server diagnostics bounded without echoing the model request."""

    try:
        body = str(response.text)
    except Exception:
        return ""
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _host_tool_messages(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Encode tool history using ordinary chat roles, never native tool metadata."""

    result: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user") or "user")
        if role == "tool":
            tool_id = str(message.get("tool_call_id", "")).strip()
            name = str(message.get("name", "")).strip()
            content = message.get("content", "")
            result.append(
                {
                    "role": "system",
                    "content": (
                        f"HOST_TOOL_RESULT id={tool_id or '-'} name={name or '-'}\n"
                        f"{content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)}"
                    ),
                }
            )
            continue

        native_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(native_calls, list) and native_calls:
            calls: list[dict[str, Any]] = []
            for index, item in enumerate(native_calls):
                if not isinstance(item, Mapping):
                    continue
                function = item.get("function")
                if not isinstance(function, Mapping):
                    continue
                raw_args = function.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception:
                    args = {}
                calls.append(
                    {
                        "id": str(item.get("id", "")).strip() or f"call_{index}",
                        "name": str(function.get("name", "")).strip(),
                        "arguments": args if isinstance(args, Mapping) else {},
                    }
                )
            result.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"kind": "tool_calls", "calls": calls},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
            continue

        content = message.get("content", "")
        result.append(
            {
                "role": role if role in {"system", "user", "assistant"} else "system",
                "content": content if isinstance(content, str) else json.dumps(
                    content,
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )

    tool_catalog = json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))
    result.append(
        {
            "role": "system",
            "content": (
                f"Host tool protocol {_HOST_TOOL_PROTOCOL_VERSION}. Native llama.cpp tool "
                "calling is disabled. Choose only from the reviewed tool catalog below. "
                "Return exactly one JSON object and no markdown. To call tools: "
                "{\"kind\":\"tool_calls\",\"calls\":[{\"id\":\"call_0\","
                "\"name\":\"TOOL_NAME\",\"arguments\":{}}]}. To finish: "
                "{\"kind\":\"final\",\"content\":\"FINAL_TEXT\"}. Multiple independent "
                "read-only calls may be emitted together. Never invent a tool name.\n"
                f"REVIEWED_TOOL_CATALOG={tool_catalog}"
            ),
        }
    )
    return result


def _parse_host_tool_envelope(
    raw: str,
    allowed_tools: frozenset[str],
    *,
    reasoning: str = "",
) -> GenerationResponse:
    if not raw:
        raise RuntimeError("Host tool envelope is empty")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise RuntimeError("Host tool envelope must be a JSON object")
    kind = str(value.get("kind", "")).strip()
    if kind == "final":
        content = value.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Host final envelope requires non-empty string content")
        return GenerationResponse(content=content.strip(), reasoning_content=reasoning)
    if kind != "tool_calls":
        raise RuntimeError("Host tool envelope kind must be 'tool_calls' or 'final'")

    raw_calls = value.get("calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise RuntimeError("Host tool_calls envelope requires a non-empty calls list")
    calls: list[ToolCall] = []
    for index, item in enumerate(raw_calls):
        if not isinstance(item, Mapping):
            raise RuntimeError("Host tool call must be an object")
        name = str(item.get("name", "")).strip()
        if not name or name not in allowed_tools:
            raise RuntimeError(f"Host tool envelope selected hidden or unknown tool {name!r}")
        arguments = item.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise RuntimeError(f"Tool {name!r} arguments must be a JSON object")
        raw_arguments = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        calls.append(
            ToolCall(
                id=str(item.get("id", "")).strip() or f"call_{index}",
                name=name,
                arguments=dict(arguments),
                raw_arguments=raw_arguments,
            )
        )
    return GenerationResponse(tool_calls=tuple(calls), reasoning_content=reasoning)


def _assert_no_native_grammar_controls(payload: Mapping[str, Any]) -> None:
    forbidden = {"tools", "tool_choice", "parallel_tool_calls", "response_format", "json_schema", "grammar"}
    present = sorted(key for key in forbidden if key in payload)
    if present:
        raise RuntimeError(
            "Host-owned llama tool protocol leaked native grammar controls: "
            + ", ".join(present)
        )


def _tool_schema_names(tools: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", "")).strip()
        if name:
            names.append(name)
    return tuple(names)


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
        raw_arguments_value = function.get("arguments", "{}")
        if isinstance(raw_arguments_value, str):
            raw_arguments = raw_arguments_value.strip() or "{}"
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Tool {name!r} returned invalid JSON arguments: {raw_arguments[:512]}"
                ) from exc
        elif isinstance(raw_arguments_value, Mapping):
            parsed = dict(raw_arguments_value)
            raw_arguments = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        else:
            raise RuntimeError(f"Tool {name!r} arguments must be a JSON object")
        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"Tool {name!r} arguments must decode to an object")
        call_id = str(item.get("id", "")).strip() or f"call_{index}"
        result.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=dict(parsed),
                raw_arguments=raw_arguments,
            )
        )
    return tuple(result)
