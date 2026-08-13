"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. The Python llama binding is intentionally not
an execution fallback: model selection, GPU residency, MTP benchmarking and metrics
all belong to the managed native llama-server process.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

from .base import (
    AdapterConfig,
    GenerationRequest,
    GenerationResponse,
    ModelAdapter,
    ModelBackendError,
    ToolCall,
)


class LlamaCppAdapter(ModelAdapter):
    """OpenAI-compatible client for the managed native llama-server."""

    _reported_server_url: str | None = None

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)

    def _server_url(self, request: GenerationRequest) -> str:
        explicit = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
        if explicit:
            return explicit
        try:
            from .. import llama_server_autotune

            _enable_jinja_tool_templates(llama_server_autotune)
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
        """Generate one OpenAI-compatible assistant turn, including tool calls."""

        cfg = self.config
        server_url = self._server_url(request)
        try:
            import httpx

            health = httpx.get(f"{server_url}/models", timeout=2.0)
            health.raise_for_status()
            if LlamaCppAdapter._reported_server_url != server_url:
                print("llama server: connected", server_url, flush=True)
                LlamaCppAdapter._reported_server_url = server_url

            # Native llama-server request compatibility has exactly one owner.
            # Tool-capable turns and ordinary text turns must use the same wire shape.
            from ..llama_server_hardware_policy import _server_payload

            payload = _server_payload(self, request)

            response = httpx.post(
                f"{server_url}/chat/completions",
                json=payload,
                timeout=None,
            )
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
        # The managed native server owns model lifetime. GPU handoff is performed by
        # llama_server_hardware_policy/llama_server_autotune, not by adapter teardown.
        return None


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


def _enable_jinja_tool_templates(autotune_module: Any) -> None:
    """Ensure every managed llama-server starts with tool-aware Jinja templates.

    llama.cpp requires ``--jinja`` for OpenAI-style function calling. Keeping this
    adaptation next to the client avoids duplicating the large server autotuner and
    applies equally to baseline and MTP variants.
    """

    base_args = getattr(autotune_module, "_base_args", None)
    if not callable(base_args) or bool(getattr(base_args, "_mmm_jinja_enabled", False)):
        return

    def tool_aware_base_args(*args: Any, **kwargs: Any) -> list[str]:
        values = list(base_args(*args, **kwargs))
        if "--jinja" not in values:
            values.append("--jinja")
        return values

    setattr(tool_aware_base_args, "_mmm_jinja_enabled", True)
    setattr(autotune_module, "_base_args", tool_aware_base_args)
