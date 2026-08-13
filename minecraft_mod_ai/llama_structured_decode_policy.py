from __future__ import annotations

from functools import wraps
from typing import Any, Mapping


_MARKER = "_mmm_bounded_section_thinking_budget_v1"


def bind_structured_decode_policy(hardware_module: Any) -> None:
    """Keep research reasoning deep but make bounded section serialization immediate.

    llama.cpp supports a per-request ``thinking_budget_tokens`` field. MMM uses zero
    only for the small, host-schema-validated ``section`` envelopes emitted after the
    research phase. Tool-capable research and all other planner calls keep their model
    default reasoning behavior.
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
            # Qwen3.5 may otherwise spend the whole completion budget in hidden
            # reasoning after the research is already complete. This is a transport
            # budget for one bounded serialization call, not a project/plan size cap.
            result["thinking_budget_tokens"] = 0
        return result

    setattr(payload, _MARKER, True)
    hardware_module._server_payload = payload


__all__ = ["bind_structured_decode_policy"]
