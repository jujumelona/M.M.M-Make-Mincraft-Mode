from __future__ import annotations

from functools import wraps
from typing import Any, Mapping


_MARKER = "_mmm_bounded_section_thinking_budget_v2"


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Keep research reasoning deep but make bounded section serialization immediate.

    llama.cpp supports per-request reasoning controls. MMM disables thinking only for
    the small, host-schema-validated ``section`` envelopes emitted after the research
    phase. Tool-capable research and all other planner calls retain model-default
    reasoning. Both controls are sent so current llama.cpp/Qwen templates terminate
    thinking immediately and existing streaming telemetry reports the state correctly.
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
        schema = getattr(request, "response_schema", None)
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        if isinstance(properties, Mapping) and "section" in properties:
            # This is a transport budget for one bounded serialization call, not a
            # project/plan-size limit. Deep reasoning already happened in research.
            result["thinking_budget_tokens"] = 0
            result["reasoning_effort"] = "none"
        return result

    setattr(payload, _MARKER, True)
    # Keep the v1 marker too so older runtime guards understand this as an upgrade.
    payload._mmm_bounded_section_thinking_budget_v1 = True  # type: ignore[attr-defined]
    hardware_module._server_payload = payload


__all__ = ["bind_structured_decode_policy"]
