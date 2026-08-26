from __future__ import annotations

import os
import sys
from functools import wraps
from typing import Any, Mapping


_MARKER = "_mmm_server_constrained_structured_decode_v1"
_RETRY_MARKER = "_mmm_structured_generation_retry_v1"


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


def _structured_response_format(request: Any) -> dict[str, Any]:
    schema = getattr(request, "response_schema", None)
    if schema is None:
        return {"type": "json_object"}
    if not isinstance(schema, Mapping):
        raise TypeError("structured response_schema must be a mapping")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_structured_response",
            "strict": True,
            "schema": dict(schema),
        },
    }


def _bind_structured_generation_retry(llama_cpp_module: Any) -> None:
    """Validate native structured text and regenerate the whole response at most once."""

    from .structured_output import (
        StructuredOutputValidationError,
        validate_structured_output,
    )

    adapter_type = llama_cpp_module.LlamaCppAdapter
    current = adapter_type.generate
    if getattr(current, _RETRY_MARKER, False):
        return

    @wraps(current)
    def generate(self: Any, request: Any) -> str:
        if (
            getattr(request, "response_format", None) != "json"
            or bool(getattr(request, "tools", ()))
        ):
            return current(self, request)

        output = current(self, request)
        try:
            return validate_structured_output(
                output,
                response_format=request.response_format,
                response_schema=request.response_schema,
            )
        except StructuredOutputValidationError as exc:
            print(
                "llama structured recovery: regenerating rejected response once",
                f" errors={len(exc.errors)}",
                file=sys.stderr,
                flush=True,
            )

        regenerated = current(self, request)
        return validate_structured_output(
            regenerated,
            response_format=request.response_format,
            response_schema=request.response_schema,
        )

    setattr(generate, _RETRY_MARKER, True)
    adapter_type.generate = generate


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Preserve native structured constraints and host validation as defense in depth.

    Every tool-free JSON request carries the same OpenAI-compatible ``response_format``
    contract used by the remote adapter. Native llama text is then host-validated after
    generation; one rejected complete response causes exactly one full regeneration.
    The malformed response itself is never extracted, coerced, patched, or repaired.
    """

    # Runtime bootstrap passes the real hardware-policy module. Keep isolated policy
    # unit tests free of process-global adapter mutation when they pass module-shaped
    # test doubles.
    if getattr(hardware_module, "__name__", "") == "minecraft_mod_ai.llama_server_hardware_policy":
        from .model_adapters import llama_cpp_adapter

        _bind_structured_generation_retry(llama_cpp_adapter)

    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        if result.get("tools"):
            return result
        if getattr(request, "response_format", None) != "json":
            return result

        result["response_format"] = _structured_response_format(request)

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


__all__ = ["_bind_structured_generation_retry", "bind_structured_decode_policy"]
