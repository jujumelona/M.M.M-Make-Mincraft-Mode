"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. Tool use follows the model's own Jinja chat
template through llama.cpp's OpenAI-compatible ``tools`` / ``tool_calls`` API.
The adapter does not impose a second model-facing JSON protocol.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

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
        """Generate one semantic turn using llama.cpp's native chat-template path.

        A reasoning-only server message is an incomplete semantic turn, not a backend
        outage. Complete it exactly once with an explicit host continuation. This is
        deliberately bounded: a second reasoning-only message fails closed instead
        of becoming an implicit retry loop.
        """

        cfg = self.config
        server_url = self._server_url(request)
        try:
            from ..llama_server_hardware_policy import _server_payload
            from ..llama_stream_efficiency_contract import _report_server_connection

            message = _completion_message(server_url, _server_payload(self, request))
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
                _server_payload(self, continuation_request),
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

    # Tool-call turns use llama.cpp's non-streaming native response so the complete
    # structured call arrives atomically. The shared client read timeout is an SSE
    # idle timeout; applying it here turns a healthy >300 s decode into ReadTimeout
    # because non-streaming responses emit no intermediate body bytes. Keep connect,
    # write and pool acquisition bounded while allowing the local decode to finish.
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
