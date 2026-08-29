from __future__ import annotations

import os
from collections.abc import Mapping
from functools import wraps
from typing import Any

from .llama_schema_transport import project_llama_transport_schema

_MARKER = "_mmm_server_constrained_structured_decode_v3"
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
    return "qwen3.5" in model_id or "qwen3.5" in filename


def _bind_structured_generation_retry(llama_cpp_module: Any) -> None:
    """Validate one generation without launching a hidden model repair turn."""

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


def _remove_native_json_constraints(payload: dict[str, Any]) -> None:
    """Remove llama.cpp sampler grammar controls while leaving host validation intact."""

    for key in ("response_format", "json_schema", "grammar"):
        payload.pop(key, None)


def _apply_llama_json_schema(
    payload: dict[str, Any],
    request: Any,
    *,
    adapter: Any | None = None,
) -> None:
    """Apply only sampler-safe transport constraints.

    Qwen3.5 is intentionally excluded from llama.cpp native JSON grammar. Its chat
    template/reasoning prefill can conflict with grammar initialization before decoding.
    The original, complete schema remains on the host request and is validated after the
    model returns.
    """

    if getattr(request, "response_format", None) != "json":
        return

    if adapter is not None and _is_qwen35(adapter):
        _remove_native_json_constraints(payload)
        return

    schema = getattr(request, "response_schema", None)
    projected = project_llama_transport_schema(schema)
    payload.pop("grammar", None)
    payload["response_format"] = {"type": "json_object"}
    if projected:
        payload["json_schema"] = projected
    else:
        payload.pop("json_schema", None)


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Separate llama.cpp transport constraints from host-owned schema validation."""

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

        qwen35 = _is_qwen35(adapter)
        _apply_llama_json_schema(result, request, adapter=adapter)

        schema = getattr(request, "response_schema", None)
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        bounded_section = isinstance(properties, Mapping) and "section" in properties

        # Qwen3.5 structured calls must not emit a reasoning prefix before host parsing.
        # This applies to research pages as well as final design sections.
        if qwen35 or schema is None or bounded_section:
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
    payload._mmm_qwen35_host_validated_json = True  # type: ignore[attr-defined]
    payload._mmm_bounded_section_thinking_budget_v2 = True  # type: ignore[attr-defined]
    payload._mmm_bounded_section_thinking_budget_v1 = True  # type: ignore[attr-defined]
    hardware_module._server_payload = payload


__all__ = [
    "_apply_llama_json_schema",
    "_bind_structured_generation_retry",
    "_is_qwen35",
    "_remove_native_json_constraints",
    "bind_structured_decode_policy",
]
