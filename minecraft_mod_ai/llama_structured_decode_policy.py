from __future__ import annotations

import os
from functools import wraps
from typing import Any, Mapping


_MARKER = "_mmm_server_constrained_structured_decode_v1"


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


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Preserve structured constraints on the native llama-server wire boundary.

    Host validation remains authoritative after generation, but it is defense in depth:
    it must not replace constrained decoding. Every tool-free JSON request therefore
    carries the same OpenAI-compatible ``response_format`` contract used by the remote
    adapter. This prevents syntactically malformed JSON from being produced merely
    because a downstream host validator exists.
    """

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
        result["reasoning_effort"] = "none"
        result["chat_template_kwargs"] = {"enable_thinking": False}
        result.pop("thinking_budget_tokens", None)

        schema = getattr(request, "response_schema", None)
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        if isinstance(properties, Mapping) and "section" in properties:
            # This is a transport budget for one explicitly bounded serialization
            # call, not a project/plan-size or input-context limit.
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


__all__ = ["bind_structured_decode_policy"]
