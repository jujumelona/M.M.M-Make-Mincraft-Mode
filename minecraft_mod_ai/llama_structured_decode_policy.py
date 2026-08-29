from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from functools import wraps
from typing import Any

_MARKER = "_mmm_server_constrained_structured_decode_v2"
_VALIDATION_MARKER = "_mmm_structured_generation_validation_v3"


def _bounded_section_output_tokens(adapter: Any) -> int:
    configured = max(1, int(getattr(adapter.config, "max_new_tokens", 1) or 1))
    raw = os.environ.get("MMM_LLAMA_BOUNDED_SECTION_MAX_TOKENS", "").strip()
    try:
        requested = int(raw) if raw else 2048
    except ValueError:
        requested = 2048
    if requested <= 0:
        requested = 2048
    return min(configured, requested)


def _is_qwen35(adapter: Any) -> bool:
    config = getattr(adapter, "config", None)
    model_id = str(getattr(config, "model_id", "")).casefold()
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", "")).casefold()
        if isinstance(extra, Mapping)
        else ""
    )
    return "qwen3.5-9b" in model_id and ("mtp" in model_id or "mtp" in filename)


def _copy_request_with(request: Any, **changes: Any) -> Any:
    """Copy the request protocol without assuming one concrete request class."""

    if is_dataclass(request) and not isinstance(request, type):
        return replace(request, **changes)
    cloned = copy.copy(request)
    for key, value in changes.items():
        setattr(cloned, key, value)
    return cloned


def _structured_repair_request(request: Any, exc: Any) -> Any:
    """Legacy compatibility helper.

    Structured planner generation no longer invokes a hidden model repair turn. Callers
    that explicitly use this helper still receive the old compact request shape.
    """

    schema = getattr(request, "response_schema", None)
    invalid_output = str(getattr(exc, "output", "") or "")
    errors = list(getattr(exc, "errors", ()) or ())
    schema_payload = dict(schema) if isinstance(schema, Mapping) else None
    repair_context = (
        "invalid_output (verbatim):\n"
        f"{invalid_output}\n"
        "validation_errors:\n"
        + json.dumps(
            errors,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\nresponse_schema:\n"
        + json.dumps(
            schema_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    messages = (
        {
            "role": "system",
            "content": (
                "Repair one already-generated JSON value. Preserve every field/value that "
                "does not violate the supplied validation errors. Change only the minimum "
                "invalid or missing fields needed to satisfy the schema. Do not redo the "
                "underlying research, planning, reasoning, or implementation. Return only "
                "the corrected JSON object."
            ),
        },
        {"role": "user", "content": repair_context},
    )
    return _copy_request_with(
        request,
        messages=messages,
        media_paths=(),
        tools=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )


def _bind_structured_generation_retry(llama_cpp_module: Any) -> None:
    """Validate one constrained generation without launching a hidden repair turn."""

    from .structured_output import validate_structured_output

    adapter_type = llama_cpp_module.LlamaCppAdapter
    current = adapter_type.generate
    if getattr(current, _VALIDATION_MARKER, False):
        return

    @wraps(current)
    def generate(self: Any, request: Any) -> str:
        output = current(self, request)
        if (
            getattr(request, "response_format", None) != "json"
            or bool(getattr(request, "tools", ()))
        ):
            return output
        return validate_structured_output(
            output,
            response_format=request.response_format,
            response_schema=request.response_schema,
        )

    setattr(generate, _VALIDATION_MARKER, True)
    generate._mmm_structured_validation_only = True  # type: ignore[attr-defined]
    adapter_type.generate = generate


def _apply_llama_json_schema(
    payload: dict[str, Any],
    request: Any,
) -> None:
    """Put the host-owned JSON schema onto llama.cpp's chat-completions payload."""

    if getattr(request, "response_format", None) != "json":
        return
    payload["response_format"] = {"type": "json_object"}
    schema = getattr(request, "response_schema", None)
    if isinstance(schema, Mapping):
        payload["json_schema"] = copy.deepcopy(dict(schema))


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Bind real server-side schema constraints and bounded JSON page budgets."""

    if (
        getattr(hardware_module, "__name__", "")
        == "minecraft_mod_ai.llama_server_hardware_policy"
    ):
        from .model_adapters import llama_cpp_adapter

        _bind_structured_generation_retry(llama_cpp_adapter)

    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = dict(current(adapter, request))
        if result.get("tools"):
            return result
        if getattr(request, "response_format", None) != "json":
            return result

        _apply_llama_json_schema(result, request)

        schema = getattr(request, "response_schema", None)
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        bounded_section = isinstance(properties, Mapping) and "section" in properties
        qwen35_game_design = (
            isinstance(properties, Mapping)
            and "game_design" in properties
            and _is_qwen35(adapter)
        )
        if schema is None or bounded_section or qwen35_game_design:
            result["reasoning_effort"] = "none"
            result["chat_template_kwargs"] = {"enable_thinking": False}
            result.pop("thinking_budget_tokens", None)

        if bounded_section:
            current_max = max(1, int(result.get("max_tokens", 1) or 1))
            result["max_tokens"] = min(
                current_max,
                _bounded_section_output_tokens(adapter),
            )
            result["thinking_budget_tokens"] = 0
        return result

    setattr(payload, _MARKER, True)
    payload._mmm_server_constrained_structured_decode = True  # type: ignore[attr-defined]
    payload._mmm_bounded_section_thinking_budget_v2 = True  # type: ignore[attr-defined]
    payload._mmm_bounded_section_thinking_budget_v1 = True  # type: ignore[attr-defined]
    hardware_module._server_payload = payload


__all__ = [
    "_apply_llama_json_schema",
    "_bind_structured_generation_retry",
    "_structured_repair_request",
    "bind_structured_decode_policy",
]
