"""Direct OpenAI-compatible llama.cpp adapter.

Tool turns use exactly one native ``/v1/chat/completions`` request. Native
``message.tool_calls`` remain authoritative. When llama-server leaves Qwen's native
``<tool_call>`` markup in ``message.content`` with no structured calls, the adapter
recovers that markup once at the response boundary and validates it against the same
host schema surface. The adapter never regenerates arguments, retries a semantic
response, or turns ordinary prose into an executable tool call.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

import httpx

from .base import (
    GenerationRequest,
    GenerationResponse,
    ModelAdapter,
    ModelBackendError,
    ToolCall,
)
from .qwen_tool_parser import ToolCallValidationError, parse_qwen_tool_markup

_DEFAULT_HTTPX_POST = httpx.post
_DEFAULT_COMPLETION_TIMEOUT_SECONDS = 120.0
_REJECTED_TOOL_CALL_NAME = "__mmm_rejected_tool_call__"
_QWEN_TOOL_CALL_OPEN = "<tool_call>"
_QWEN_FUNCTION_OPEN = "<function="
_QWEN_PAYLOAD_PARAMETER_OPEN = "<parameter=payload>"
_QWEN_ARGUMENTS_PARAMETER_OPEN = "<parameter=arguments>"


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

    def input_context_accounting(self, request: GenerationRequest):
        """Ask the live llama.cpp server to count the exact templated request."""

        from ..llama_exact_context import live_context_accounting
        from ..llama_server_hardware_policy import _server_payload

        server_url = self._server_url(request)
        return live_context_accounting(server_url, _server_payload(self, request))

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
        """Generate exactly one assistant turn from exactly one completion request."""

        server_url = self._server_url(request)
        try:
            if request.tools:
                return _native_tool_completion(self, server_url, request)
            return _plain_completion(self, server_url, request)
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc

    def close(self) -> None:
        return None


def _normalize_tool_schema(tool: Any) -> Mapping[str, Any]:
    schema = tool.to_schema() if hasattr(tool, "to_schema") else tool
    if not isinstance(schema, Mapping):
        raise TypeError("tool definition must be a mapping or expose to_schema()")
    return deepcopy(dict(schema))


def _normalized_tool_request(request: GenerationRequest) -> GenerationRequest:
    tools = tuple(_normalize_tool_schema(tool) for tool in request.tools)
    validation = tuple(
        _normalize_tool_schema(tool) for tool in request.tool_validation_schemas
    )
    return replace(
        request,
        tools=tools,
        tool_validation_schemas=validation,
    )


def _plain_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    from ..llama_exact_context import capacity_safe_payload
    from ..llama_server_hardware_policy import _server_payload
    from ..llama_stream_efficiency_contract import _report_server_connection

    payload = capacity_safe_payload(
        server_url,
        _server_payload(adapter, request),
        structured_output=request.response_format == "json",
    )
    message = _completion_message(server_url, payload)
    _report_server_connection(server_url)
    if message.get("tool_calls"):
        raise RuntimeError("plain completion unexpectedly returned tool_calls")
    content = message.get("content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    content_text = content if isinstance(content, str) else ""
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    if not content_text.strip() and not reasoning_text.strip():
        raise RuntimeError("native llama-server returned an empty assistant message")
    return GenerationResponse(
        content=content_text.strip(),
        reasoning_content=reasoning_text.strip(),
    )


def _native_tool_completion(
    adapter: LlamaCppAdapter,
    server_url: str,
    request: GenerationRequest,
) -> GenerationResponse:
    """Run one native tool completion without semantic regeneration or whole-turn repair."""

    from ..llama_exact_context import capacity_safe_payload
    from ..llama_stream_efficiency_contract import _report_server_connection

    request = _normalized_tool_request(request)
    payload = capacity_safe_payload(
        server_url,
        _tool_server_payload(adapter, request),
        structured_output=False,
    )
    message = _completion_message(server_url, payload)
    _report_server_connection(server_url)
    return _native_tool_generation_response(message, request)


def _tool_choice_requires_call(choice: Any) -> bool:
    if isinstance(choice, Mapping):
        return True
    if not isinstance(choice, str):
        return False
    return choice.strip().casefold() not in {"", "auto", "none"}


def _contains_qwen_tool_markup(text: str) -> bool:
    return _QWEN_TOOL_CALL_OPEN in text or text.lstrip().startswith(_QWEN_FUNCTION_OPEN)


def _qwen_markup_for_schema(
    text: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> str:
    """Map Qwen's generic payload container only when no tool owns a payload field."""

    if _QWEN_PAYLOAD_PARAMETER_OPEN not in text:
        return text
    for schema in schemas.values():
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping) and "payload" in properties:
            return text
    return text.replace(
        _QWEN_PAYLOAD_PARAMETER_OPEN,
        _QWEN_ARGUMENTS_PARAMETER_OPEN,
    )


def _native_tool_generation_response(
    message: Mapping[str, Any],
    request: GenerationRequest,
) -> GenerationResponse:
    """Decode one assistant message while preserving the required-tool contract."""

    request = _normalized_tool_request(request)
    schemas = _request_tool_schema_map(request)
    raw_calls = _raw_native_tool_calls(message)
    content = message.get("content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    content_text = content if isinstance(content, str) else ""
    reasoning_text = reasoning if isinstance(reasoning, str) else ""

    parse_rejections: tuple[ToolCall, ...] = ()
    parsed: tuple[ToolCall, ...]
    if raw_calls:
        if not request.parallel_tool_calls and len(raw_calls) > 1:
            raise ToolCallValidationError(
                "model emitted parallel tool calls when they are disabled"
            )
        parsed, parse_rejections = _parse_native_tool_calls_isolated(raw_calls)
    elif _contains_qwen_tool_markup(content_text):
        content_text, parsed = parse_qwen_tool_markup(
            _qwen_markup_for_schema(content_text, schemas),
            schemas,
        )
        if not request.parallel_tool_calls and len(parsed) > 1:
            raise ToolCallValidationError(
                "model emitted parallel tool calls when they are disabled"
            )
    else:
        parsed = ()

    valid_calls, schema_rejections = _partition_tool_calls_against_host_schema(
        parsed, schemas
    )
    rejections = (*parse_rejections, *schema_rejections)

    if not valid_calls and rejections:
        return GenerationResponse(
            content=content_text.strip(),
            tool_calls=rejections,
            reasoning_content=reasoning_text.strip(),
        )
    if not valid_calls and _tool_choice_requires_call(request.tool_choice):
        raise ToolCallValidationError(
            "model omitted required native tool call"
        )

    return GenerationResponse(
        content=content_text.strip(),
        tool_calls=(*valid_calls, *rejections),
        reasoning_content=reasoning_text.strip(),
    )


def _tool_definition_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise RuntimeError("native tool transport exposed a schema without function metadata")
    name = str(function.get("name", "")).strip()
    if not name:
        raise RuntimeError("native tool transport exposed an unnamed function schema")
    return name


def _exact_model_visible_tools(
    declared: Sequence[Mapping[str, Any]],
    transported: Any,
) -> list[Mapping[str, Any]]:
    """Restore host-declared schemas for exactly the tool names selected by transport."""

    if not isinstance(transported, Sequence) or isinstance(
        transported, (str, bytes, bytearray)
    ):
        raise RuntimeError("native llama tool transport dropped the tools surface")

    declared_by_name: dict[str, Mapping[str, Any]] = {}
    for tool in declared:
        name = _tool_definition_name(tool)
        if name in declared_by_name:
            raise RuntimeError(f"duplicate model-visible tool schema {name!r}")
        declared_by_name[name] = tool

    exact: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in transported:
        if not isinstance(raw, Mapping):
            raise RuntimeError("native llama tool transport exposed a non-object schema")
        name = _tool_definition_name(raw)
        if name not in declared_by_name:
            raise RuntimeError(f"native llama transport exposed undeclared tool {name!r}")
        if name in seen:
            raise RuntimeError(f"native llama transport duplicated tool schema {name!r}")
        seen.add(name)
        exact.append(deepcopy(dict(declared_by_name[name])))
    return exact


def _tool_server_payload(
    adapter: LlamaCppAdapter,
    request: GenerationRequest,
) -> dict[str, Any]:
    from ..llama_server_hardware_policy import _server_payload, _server_tool_choice

    payload = _server_payload(adapter, request)
    transported = payload.get("tools")
    payload["tools"] = _exact_model_visible_tools(request.tools, transported)
    if not payload["tools"]:
        raise RuntimeError("native tool transport received no tool schemas")
    expected = _server_tool_choice(request)
    if payload.get("tool_choice") != expected:
        raise RuntimeError(
            "llama hardware policy violated native tool transport: "
            f"tool_choice must be {expected!r}"
        )
    return payload


def _request_tool_schema_map(request: GenerationRequest) -> dict[str, Mapping[str, Any]]:
    """Return the parse/validation surface without widening model visibility."""

    from ..tool_validation_surface_contract import _validation_surface

    visible = tuple(request.tools)
    authorized = tuple(request.tool_validation_schemas)
    validation_surface = _validation_surface(visible, authorized)
    return _tool_schema_map(tuple(dict(tool) for tool in validation_surface))


_request_tool_schema_map._mmm_core_validation_surface = True  # type: ignore[attr-defined]


def _tool_schema_map(
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for schema in schemas:
        function = schema.get("function")
        if not isinstance(function, Mapping):
            raise TypeError("tool schema lacks function metadata")
        name = str(function.get("name", "")).strip()
        if not name:
            raise ToolCallValidationError("tool schema lacks a function name")
        if name in result:
            raise ToolCallValidationError(f"duplicate tool schema name {name!r}")
        parameters = function.get("parameters", {})
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ToolCallValidationError(
                f"tool {name!r} parameters schema must be an object"
            )
        result[name] = dict(parameters or {})
    return result


def _raw_native_tool_calls(message: Mapping[str, Any]) -> list[Any]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise ToolCallValidationError(
            "llama-server returned tool_calls in a non-list shape"
        )
    return raw_calls


def _parse_native_tool_call(raw_call: Any, *, index: int) -> ToolCall:
    if not isinstance(raw_call, Mapping):
        raise ToolCallValidationError(
            "llama-server returned an invalid structured tool call"
        )
    call_type = str(raw_call.get("type", "function") or "function").strip()
    if call_type != "function":
        raise ToolCallValidationError(
            f"llama-server returned unsupported tool call type {call_type!r}"
        )
    function = raw_call.get("function")
    if not isinstance(function, Mapping):
        raise ToolCallValidationError(
            "llama-server structured tool call lacks function metadata"
        )
    name = str(function.get("name", "")).strip()
    if not name:
        raise ToolCallValidationError(
            "llama-server structured tool call has an empty function name"
        )
    raw_value = function.get("arguments", "{}")
    if isinstance(raw_value, Mapping):
        arguments = dict(raw_value)
        raw_arguments = json.dumps(
            arguments, ensure_ascii=False, separators=(",", ":")
        )
    elif isinstance(raw_value, str):
        raw_arguments = raw_value
        try:
            decoded = json.loads(raw_value.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise ToolCallValidationError(
                f"llama-server structured tool {name!r} returned invalid JSON arguments"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ToolCallValidationError(
                f"llama-server structured tool {name!r} arguments must be an object"
            )
        arguments = dict(decoded)
    else:
        raise ToolCallValidationError(
            f"llama-server structured tool {name!r} arguments have an invalid shape"
        )
    call_id = str(raw_call.get("id", "")).strip()
    if not call_id:
        digest = hashlib.sha256(
            f"{index}\0{name}\0{raw_arguments}".encode()
        ).hexdigest()[:16]
        call_id = f"call_{digest}"
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def _parse_native_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """Strict compatibility parser used by direct parser contracts."""

    return tuple(
        _parse_native_tool_call(raw_call, index=index)
        for index, raw_call in enumerate(_raw_native_tool_calls(message))
    )


def _raw_call_metadata(raw_call: Any) -> tuple[str, str]:
    if not isinstance(raw_call, Mapping):
        return "", ""
    function = raw_call.get("function")
    if not isinstance(function, Mapping):
        return "", ""
    name = str(function.get("name", "") or "").strip()
    raw_value = function.get("arguments", "")
    if isinstance(raw_value, str):
        raw_arguments = raw_value
    elif isinstance(raw_value, Mapping):
        raw_arguments = json.dumps(
            dict(raw_value), ensure_ascii=False, separators=(",", ":")
        )
    else:
        raw_arguments = repr(raw_value)
    return name, raw_arguments


def _rejected_tool_call(
    *,
    index: int,
    original_tool: str,
    raw_arguments: str,
    failure_code: str,
    error: str,
) -> ToolCall:
    rejection = {
        "original_tool": original_tool,
        "failure_code": failure_code,
        "error": error,
        "raw_arguments": raw_arguments,
    }
    serialized = json.dumps(
        rejection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{index}\0{original_tool}\0{raw_arguments}\0{failure_code}\0{error}".encode()
    ).hexdigest()[:16]
    return ToolCall(
        id=f"rejected_{digest}",
        name=_REJECTED_TOOL_CALL_NAME,
        arguments=rejection,
        raw_arguments=serialized,
    )


def _parse_native_tool_calls_isolated(
    raw_calls: Sequence[Any],
) -> tuple[tuple[ToolCall, ...], tuple[ToolCall, ...]]:
    accepted: list[ToolCall] = []
    rejected: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        try:
            accepted.append(_parse_native_tool_call(raw_call, index=index))
        except ToolCallValidationError as exc:
            original_tool, raw_arguments = _raw_call_metadata(raw_call)
            message = str(exc)
            failure_code = (
                "TOOL_ARGUMENT_JSON_INVALID"
                if "invalid JSON arguments" in message
                else "TOOL_CALL_MALFORMED"
            )
            rejected.append(
                _rejected_tool_call(
                    index=index,
                    original_tool=original_tool,
                    raw_arguments=raw_arguments,
                    failure_code=failure_code,
                    error=message,
                )
            )
    return tuple(accepted), tuple(rejected)


def _validate_tool_call_against_host_schema(
    call: ToolCall,
    schemas: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        from jsonschema.validators import validator_for
    except Exception as exc:
        raise RuntimeError("host tool schema validation is unavailable") from exc

    schema = schemas.get(call.name)
    if schema is None:
        raise ToolCallValidationError(
            f"model emitted non-visible tool {call.name!r}"
        )
    try:
        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
        errors = sorted(
            validator_type(schema).iter_errors(dict(call.arguments)),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        raise RuntimeError(
            f"tool {call.name!r} has an invalid host validation schema"
        ) from exc
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path)
    detail = " ".join(str(error.message).split())[:240]
    location = f" at {path}" if path else ""
    raise ToolCallValidationError(
        f"tool {call.name!r} emitted schema-invalid arguments{location}: {detail}"
    )


def _validate_tool_calls_against_host_schema(
    calls: Sequence[ToolCall],
    schemas: Mapping[str, Mapping[str, Any]],
) -> None:
    """Strict compatibility validator used by direct validation contracts."""

    for call in calls:
        _validate_tool_call_against_host_schema(call, schemas)


def _partition_tool_calls_against_host_schema(
    calls: Sequence[ToolCall],
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[ToolCall, ...], tuple[ToolCall, ...]]:
    accepted: list[ToolCall] = []
    rejected: list[ToolCall] = []
    for index, call in enumerate(calls):
        try:
            _validate_tool_call_against_host_schema(call, schemas)
        except ToolCallValidationError as exc:
            message = str(exc)
            failure_code = (
                "TOOL_NOT_VISIBLE"
                if "non-visible tool" in message
                else "TOOL_SCHEMA_INVALID"
            )
            rejected.append(
                _rejected_tool_call(
                    index=index,
                    original_tool=call.name,
                    raw_arguments=call.raw_arguments,
                    failure_code=failure_code,
                    error=message,
                )
            )
        else:
            accepted.append(call)
    return tuple(accepted), tuple(rejected)


def _completion_message(server_url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    from ..llama_finish_reason_contract import (
        CONTEXT_PRESSURE,
        LlamaCompletionBoundaryError,
        _http_context_pressure,
        _length_error,
    )

    response = _post_completion(server_url, payload)
    if response.status_code >= 400:
        body = _bounded_response_body(response)
        if _http_context_pressure(response.status_code, body):
            raise LlamaCompletionBoundaryError(
                f"llama-server rejected prompt context: {body}", kind=CONTEXT_PRESSURE,
            )
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
        raise TypeError("native llama-server returned an invalid completion choice")
    finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        raise _length_error(data, payload)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise TypeError("native llama-server returned no assistant message")
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
    timeout = httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)
    try:
        if httpx.post is not _DEFAULT_HTTPX_POST:
            return httpx.post(endpoint, json=dict(payload), timeout=timeout)
        from ..llama_stream_efficiency_contract import _client

        return _client(server_url).post(endpoint, json=dict(payload), timeout=timeout)
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "native llama-server completion made no readable progress for "
            f"{read_timeout:.0f}s"
        ) from exc


def _bounded_response_body(response: Any, *, limit: int = 1600) -> str:
    try:
        body = str(response.text)
    except (AttributeError, httpx.HTTPError, RuntimeError, TypeError, ValueError):
        return ""
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."
