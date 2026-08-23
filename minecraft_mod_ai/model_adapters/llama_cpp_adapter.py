"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only. Tool-aware Qwen turns keep the model's raw Jinja
markup as data and let MMM's host parser own tool recognition, schema validation, and
execution authority. Server-parsed OpenAI ``message.tool_calls`` are rejected so a
transport/parser change cannot bypass the same host validation boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import httpx

from .base import (
    GenerationRequest,
    GenerationResponse,
    ModelAdapter,
    ModelBackendError,
    ToolCall,
)


_DEFAULT_HTTPX_POST = httpx.post
_DEFAULT_COMPLETION_TIMEOUT_SECONDS = 600.0
_DEFAULT_COMPLETION_HEARTBEAT_SECONDS = 15.0
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
_PREFILL_CALIBRATION_SENTINEL = "MMM_ASSISTANT_PREFILL_CALIBRATION_V1"
_MAX_PREFILL_TEMPLATE_BYTES = 512
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
        """Generate one semantic assistant turn."""

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


def _plain_semantic_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    from ..llama_server_hardware_policy import _server_payload
    from ..llama_stream_efficiency_contract import _report_server_connection

    message = _completion_message_with_prefill(
        adapter,
        server_url,
        _server_payload(adapter, request),
    )
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
    continued_message = _completion_message_with_prefill(
        adapter,
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
    """Return one host-validated tool/content action, with one reasoning continuation."""

    from ..llama_stream_efficiency_contract import _report_server_connection

    message = _completion_message_with_prefill(
        adapter,
        server_url,
        _tool_server_payload(adapter, request),
    )
    _report_server_connection(server_url)
    turn = _qwen_tool_generation_response(message, request)
    if _has_semantic_action(turn):
        return turn
    if not turn.reasoning_content:
        raise RuntimeError(
            "native llama-server returned neither visible content, reasoning, nor Qwen tool calls"
        )

    continuation_request = _reasoning_continuation_request(
        request,
        turn.reasoning_content,
    )
    continued_message = _completion_message_with_prefill(
        adapter,
        server_url,
        _tool_server_payload(adapter, continuation_request),
    )
    continued = _qwen_tool_generation_response(continued_message, continuation_request)
    if not _has_semantic_action(continued):
        if continued.reasoning_content:
            raise RuntimeError(
                "native llama-server returned a reasoning-only tool continuation without a semantic action"
            )
        raise RuntimeError(
            "native llama-server returned no semantic action after a reasoning-only tool continuation"
        )
    return GenerationResponse(
        content=continued.content,
        tool_calls=continued.tool_calls,
        reasoning_content=_merge_reasoning(
            turn.reasoning_content,
            continued.reasoning_content,
        ),
    )


def _tool_server_payload(
    adapter: LlamaCppAdapter,
    request: GenerationRequest,
) -> dict[str, Any]:
    """Assert the canonical hardware policy preserves native Jinja tool semantics."""

    from ..llama_server_hardware_policy import _server_payload, _server_tool_choice

    payload = _server_payload(adapter, request)
    if not payload.get("tools"):
        raise RuntimeError("native tool transport received no tool schemas")
    expected_choice = _server_tool_choice(request)
    if payload.get("tool_choice") != expected_choice:
        raise RuntimeError(
            "llama hardware policy violated native tool transport: "
            f"tool_choice must be {expected_choice!r}"
        )
    return payload


def _merge_text_progress(previous: Any, current: Any) -> str:
    first = previous if isinstance(previous, str) else ""
    second = current if isinstance(current, str) else ""
    return first + second


def _merge_partial_messages(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(previous or {})
    incoming = dict(current or {})
    result["role"] = "assistant"
    for key in ("reasoning_content", "reasoning", "content"):
        if key in result or key in incoming:
            result[key] = _merge_text_progress(result.get(key), incoming.get(key))
    for key, value in incoming.items():
        if key not in {"role", "reasoning_content", "reasoning", "content"}:
            result[key] = value
    return result


def _reject_partial_server_tool_calls(message: Mapping[str, Any]) -> None:
    if message.get("tool_calls"):
        raise RuntimeError(
            "llama-server returned server-parsed tool_calls inside an incomplete response; "
            "partial tool actions are never executable"
        )


def _assistant_prefill_payload(
    original: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(original)
    messages = [dict(message) for message in original.get("messages", ())]
    assistant = {
        key: value
        for key, value in generated.items()
        if key in {"role", "content", "reasoning_content", "reasoning"}
    }
    assistant["role"] = "assistant"
    if messages and messages[-1].get("role") == "assistant":
        messages[-1] = _merge_partial_messages(messages[-1], assistant)
    else:
        messages.append(assistant)
    payload["messages"] = messages
    return payload


def _normalize_assistant_prefill_suffix(
    message: Mapping[str, Any],
    *,
    continuation_page: bool,
    template_prefix: str,
) -> dict[str, Any]:
    result = dict(message)
    content = result.get("content")
    if not continuation_page or not template_prefix:
        return result
    if not isinstance(content, str) or not content.startswith(template_prefix):
        raise RuntimeError(
            "live llama-server assistant-prefill prefix changed after calibration"
        )
    result["content"] = content[len(template_prefix) :]
    return result


def _assistant_prefill_calibration_payload(
    original: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": original.get("model", "local"),
        "messages": [
            {
                "role": "user",
                "content": "Calibrate the trailing-assistant template. Generate no tokens.",
            },
            {"role": "assistant", "content": _PREFILL_CALIBRATION_SENTINEL},
        ],
        "max_tokens": 0,
        "temperature": 0.0,
    }
    for key in (
        "chat_template_kwargs",
        "reasoning_effort",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    ):
        if key in original:
            payload[key] = original[key]
    return payload


def _calibrate_assistant_prefill_generation_prompt(
    server_url: str,
    original: Mapping[str, Any],
) -> str:
    response = _post_completion(
        server_url,
        _assistant_prefill_calibration_payload(original),
    )
    if response.status_code >= 400:
        raise RuntimeError("assistant-prefill calibration request was rejected")
    data = response.json()
    if not isinstance(data, Mapping):
        raise RuntimeError("assistant-prefill calibration returned invalid JSON")
    usage = data.get("usage")
    if not isinstance(usage, Mapping) or usage.get("completion_tokens") != 0:
        raise RuntimeError("assistant-prefill calibration generated model tokens")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("assistant-prefill calibration returned invalid choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("assistant-prefill calibration returned an invalid choice")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("assistant-prefill calibration returned no message")
    if message.get("tool_calls"):
        raise RuntimeError("assistant-prefill calibration returned a tool call")
    if message.get("reasoning_content") or message.get("reasoning"):
        raise RuntimeError("assistant-prefill calibration returned reasoning")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise RuntimeError("assistant-prefill calibration prefix is empty or ambiguous")
    encoded = content.encode("utf-8")
    if len(encoded) > _MAX_PREFILL_TEMPLATE_BYTES:
        raise RuntimeError("assistant-prefill calibration prefix is unexpectedly large")
    if _PREFILL_CALIBRATION_SENTINEL in content:
        raise RuntimeError("assistant-prefill calibration echoed the supplied prefill")
    return content


def _completion_message_with_prefill(
    adapter: LlamaCppAdapter,
    server_url: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    from ..llama_finish_reason_contract import (
        CONTEXT_PRESSURE,
        LlamaCompletionBoundaryError,
        OUTPUT_EXHAUSTED,
        _CONTEXT_ERROR,
        _OUTPUT_ERROR,
        completion_boundary_error,
        partial_message_receipt,
    )
    from ..qwen_family_capabilities import qwen_family_capabilities

    extra = getattr(getattr(adapter, "config", None), "extra", {})
    qwen_thinking_page = False
    qwen_nonthinking_page = False
    if isinstance(extra, Mapping) and str(
        extra.get("runtime_contract", "")
    ).strip().casefold() == "qwen":
        capabilities = qwen_family_capabilities(adapter.config, required=True)
        if capabilities is None or not capabilities.assistant_prefill:
            raise RuntimeError(
                "registry Qwen family does not permit assistant-prefill continuation"
            )
        template_kwargs = payload.get("chat_template_kwargs")
        qwen_thinking_page = not (
            isinstance(template_kwargs, Mapping)
            and template_kwargs.get("enable_thinking") is False
        )
        qwen_nonthinking_page = not qwen_thinking_page

    original_payload = dict(payload)
    current_payload = original_payload
    accumulated: dict[str, Any] = {}
    progress_bytes = 0
    progress_sha256 = ""
    calibrated_template_prefix = ""

    while True:
        try:
            final_message = _completion_message(server_url, current_payload)
        except RuntimeError as exc:
            boundary = completion_boundary_error(exc)
            if boundary is None:
                raise
            try:
                partial = _normalize_assistant_prefill_suffix(
                    boundary.partial_message,
                    continuation_page=bool(accumulated),
                    template_prefix=calibrated_template_prefix,
                )
            except RuntimeError as prefix_exc:
                if not accumulated:
                    raise
                message = _CONTEXT_ERROR if boundary.kind == CONTEXT_PRESSURE else _OUTPUT_ERROR
                raise LlamaCompletionBoundaryError(
                    message
                    + "; live assistant-prefill normalization changed;"
                    + f" preserved_partial_bytes={partial_message_receipt(accumulated)[0]}",
                    kind=boundary.kind,
                    partial_message=accumulated,
                    prompt_tokens=boundary.prompt_tokens,
                    completion_tokens=boundary.completion_tokens,
                    max_tokens=boundary.max_tokens,
                ) from prefix_exc
            _reject_partial_server_tool_calls(partial)
            merged = _merge_partial_messages(accumulated, partial)
            if boundary.kind == CONTEXT_PRESSURE:
                raise LlamaCompletionBoundaryError(
                    _CONTEXT_ERROR
                    + "; assistant-prefill reached the live context boundary;"
                    + f" partial_bytes={partial_message_receipt(merged)[0]}",
                    kind=CONTEXT_PRESSURE,
                    partial_message=merged,
                    prompt_tokens=boundary.prompt_tokens,
                    completion_tokens=boundary.completion_tokens,
                    max_tokens=boundary.max_tokens,
                ) from exc
            if boundary.kind != OUTPUT_EXHAUSTED:
                raise
            if qwen_thinking_page:
                raise
            next_bytes, next_sha256 = partial_message_receipt(merged)
            if next_bytes <= progress_bytes or next_sha256 == progress_sha256:
                raise LlamaCompletionBoundaryError(
                    _OUTPUT_ERROR
                    + "; assistant-prefill made no additional byte progress;"
                    + f" partial_bytes={next_bytes}",
                    kind=OUTPUT_EXHAUSTED,
                    partial_message=merged,
                    prompt_tokens=boundary.prompt_tokens,
                    completion_tokens=boundary.completion_tokens,
                    max_tokens=boundary.max_tokens,
                ) from exc
            accumulated = merged
            progress_bytes = next_bytes
            progress_sha256 = next_sha256
            if qwen_nonthinking_page and not calibrated_template_prefix:
                try:
                    calibrated_template_prefix = (
                        _calibrate_assistant_prefill_generation_prompt(
                            server_url,
                            original_payload,
                        )
                    )
                except Exception as calibration_exc:
                    raise LlamaCompletionBoundaryError(
                        _OUTPUT_ERROR
                        + "; live assistant-prefill calibration was unavailable;"
                        + f" preserved_partial_bytes={next_bytes}",
                        kind=OUTPUT_EXHAUSTED,
                        partial_message=merged,
                        prompt_tokens=boundary.prompt_tokens,
                        completion_tokens=boundary.completion_tokens,
                        max_tokens=boundary.max_tokens,
                    ) from calibration_exc
            current_payload = _assistant_prefill_payload(original_payload, accumulated)
            continue

        try:
            normalized_final = _normalize_assistant_prefill_suffix(
                final_message,
                continuation_page=bool(accumulated),
                template_prefix=calibrated_template_prefix,
            )
        except RuntimeError as prefix_exc:
            if not accumulated:
                raise
            raise LlamaCompletionBoundaryError(
                _OUTPUT_ERROR
                + "; live assistant-prefill normalization changed;"
                + f" preserved_partial_bytes={partial_message_receipt(accumulated)[0]}",
                kind=OUTPUT_EXHAUSTED,
                partial_message=accumulated,
                max_tokens=int(original_payload.get("max_tokens", 0) or 0),
            ) from prefix_exc
        if not accumulated:
            return normalized_final
        return _merge_partial_messages(accumulated, normalized_final)


def _plain_generation_response(message: Mapping[str, Any]) -> GenerationResponse:
    if message.get("tool_calls"):
        raise RuntimeError("plain completion unexpectedly returned tool_calls")
    content_value = message.get("content")
    content_raw = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    server_reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    embedded_reasoning, content = _split_qwen_reasoning_markup(content_raw)
    reasoning = _merge_reasoning(server_reasoning, embedded_reasoning)
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
            "native llama-server returned server-parsed tool_calls; Qwen tool actions "
            "must remain raw content until the MMM host parser validates them"
        )
    schemas = _tool_schema_map(request.tools)
    content_value = message.get("content")
    content_raw = content_value if isinstance(content_value, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning"))
    server_reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    embedded_reasoning, content_raw = _split_qwen_reasoning_markup(content_raw)
    reasoning_raw = _merge_reasoning(server_reasoning, embedded_reasoning)

    reasoning, reasoning_calls = _parse_qwen_tool_markup(reasoning_raw, schemas)
    content, content_calls = _parse_qwen_tool_markup(content_raw, schemas)
    calls = (*reasoning_calls, *content_calls)
    _validate_tool_choice(request, calls)
    return GenerationResponse(
        content=content.strip(),
        tool_calls=tuple(calls),
        reasoning_content=reasoning.strip(),
    )


def _split_qwen_reasoning_markup(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return "", text
    reasoning_start = len("<think>")
    reasoning_end = stripped.find("</think>", reasoning_start)
    if reasoning_end < 0:
        raise RuntimeError("Qwen reasoning block is missing </think>")
    reasoning = stripped[reasoning_start:reasoning_end].strip()
    content = stripped[reasoning_end + len("</think>") :].lstrip()
    return reasoning, content


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
    if isinstance(required_value, Sequence) and not isinstance(required_value, (str, bytes)):
        required = {str(value) for value in required_value}
    additional = schema.get("additionalProperties", True)

    arguments: dict[str, Any] = {}
    argument_sources: dict[str, str] = {}
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
        emitted_key = text[key_start:key_end].strip()
        if not emitted_key:
            raise RuntimeError(f"Qwen tool {name!r} emitted an empty parameter name")
        key = emitted_key
        if (
            name == "apply_source_edit"
            and emitted_key == "action"
            and "operation" in properties
            and "action" not in properties
        ):
            key = "operation"
        elif (
            name == "apply_source_edit"
            and emitted_key == "file"
            and "path" in properties
            and "file" not in properties
        ):
            key = "path"
        if key in arguments:
            previous = argument_sources[key]
            if {previous, emitted_key} == {"action", "operation"}:
                raise RuntimeError(
                    "Qwen tool 'apply_source_edit' emitted conflicting sources for "
                    "parameter 'operation': alias 'action' and canonical 'operation'"
                )
            if {previous, emitted_key} == {"file", "path"}:
                raise RuntimeError(
                    "Qwen tool 'apply_source_edit' emitted conflicting sources for "
                    "parameter 'path': alias 'file' and canonical 'path'"
                )
            raise RuntimeError(f"Qwen tool {name!r} repeated parameter {emitted_key!r}")
        if key not in properties and additional is False:
            raise RuntimeError(
                f"Qwen tool {name!r} emitted unknown parameter {emitted_key!r}"
            )
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
        argument_sources[key] = emitted_key
        pos = close_at + len(_PARAMETER_CLOSE)

    missing = sorted(required - arguments.keys())
    if missing:
        raise RuntimeError(
            f"Qwen tool {name!r} omitted required parameters: {', '.join(missing)}"
        )
    raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
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
            if compact.startswith(("{", "[", '"')) or compact in {"true", "false", "null"}:
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


def _validate_tool_choice(request: GenerationRequest, calls: Sequence[ToolCall]) -> None:
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
    return replace(request, messages=tuple(messages), media_paths=())


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


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _payload_content_chars(payload: Mapping[str, Any]) -> int:
    total = 0
    messages = payload.get("messages", ())
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            try:
                total += len(json.dumps(content, ensure_ascii=False))
            except (TypeError, ValueError):
                continue
    return total


def _post_completion(server_url: str, payload: Mapping[str, Any]) -> Any:
    endpoint = f"{server_url}/chat/completions"
    read_timeout = _positive_env_float(
        "MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS",
        _DEFAULT_COMPLETION_TIMEOUT_SECONDS,
    )
    heartbeat_seconds = _positive_env_float(
        "MMM_LLAMA_COMPLETION_HEARTBEAT_SECONDS",
        _DEFAULT_COMPLETION_HEARTBEAT_SECONDS,
    )
    started = time.monotonic()
    stop = threading.Event()
    input_chars = _payload_content_chars(payload)
    max_tokens = payload.get("max_tokens", "?")
    tool_count = len(payload.get("tools", ()) or ())
    print(
        "llama server: completion request",
        f" input_chars={input_chars}",
        f" max_tokens={max_tokens}",
        f" tools={tool_count}",
        f" read_timeout={read_timeout:.0f}s",
        sep="",
        flush=True,
    )

    def report_pending() -> None:
        while not stop.wait(heartbeat_seconds):
            print(
                "llama server: completion pending",
                f" elapsed={time.monotonic() - started:.1f}s",
                f" input_chars={input_chars}",
                f" max_tokens={max_tokens}",
                f" tools={tool_count}",
                sep="",
                flush=True,
            )

    reporter = threading.Thread(
        target=report_pending,
        name="mmm-llama-completion-liveness",
        daemon=True,
    )
    reporter.start()
    timeout = httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)
    try:
        if httpx.post is not _DEFAULT_HTTPX_POST:
            return httpx.post(endpoint, json=payload, timeout=timeout)
        from ..llama_stream_efficiency_contract import _client

        return _client(server_url).post(endpoint, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "native llama-server completion made no readable progress for "
            f"{read_timeout:.0f}s"
        ) from exc
    finally:
        stop.set()
        reporter.join(timeout=0.2)


def _bounded_response_body(response: Any, *, limit: int = 1600) -> str:
    try:
        body = str(response.text)
    except Exception:
        return ""
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."
