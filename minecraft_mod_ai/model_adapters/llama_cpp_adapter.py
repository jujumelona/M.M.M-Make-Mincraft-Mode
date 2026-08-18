"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. Tool use follows the model's own Jinja chat
template through llama.cpp's OpenAI-compatible ``tools`` / ``tool_calls`` API.
The adapter does not impose a second model-facing JSON protocol.
"""
from __future__ import annotations

import json
import os
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


class LlamaCppAdapter(ModelAdapter):
    """OpenAI-compatible client for the managed native llama-server."""

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
        """Generate one turn using llama.cpp's native chat-template tool path."""

        cfg = self.config
        server_url = self._server_url(request)
        try:
            from ..llama_server_hardware_policy import _server_payload
            from ..llama_stream_efficiency_contract import _report_server_connection

            message = _completion_message(server_url, _server_payload(self, request))
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
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("native llama-server returned an invalid completion choice")
    finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        raise RuntimeError(
            "native llama-server truncated the completion at max_tokens; paginate the "
            "response or raise the explicit Qwen output limit instead of repairing a "
            "partial answer"
        )
    message = choice.get("message")
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
