from __future__ import annotations

"""Pure user-intent projection shared by tool retrieval and execution policy."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

_INTENT_FIELDS = (
    "phase",
    "task",
    "goal",
    "objective",
    "action",
    "request",
    "query",
    "instruction",
)
_MODULE_FIELDS = ("id", "module_id", "name", "kind", "type")
_MAX_ROUTING_CHARS = 12_000


def _payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip().startswith("{"):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _render_payload(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field in _INTENT_FIELDS:
        value = payload.get(field)
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            parts.append(f"{field}: {value}")

    module = payload.get("module")
    if isinstance(module, Mapping):
        for field in _MODULE_FIELDS:
            value = module.get(field)
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                parts.append(f"module.{field}: {value}")

    rules = payload.get("rules")
    if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes, bytearray)):
        for value in rules[:32]:
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                parts.append(f"rule: {value}")
    return "\n".join(parts)


def structured_user_intent(
    messages: Sequence[Mapping[str, Any]],
    *,
    limit: int = _MAX_ROUTING_CHARS,
) -> str:
    """Return a bounded routing query containing user intent, not tool-history noise."""

    parts: list[str] = []
    size = 0
    for message in reversed(messages):
        if str(message.get("role", "")).strip().casefold() != "user":
            continue
        content = message.get("content")
        payload = _payload(content)
        if payload is not None:
            rendered = _render_payload(payload)
            if not rendered and isinstance(content, str):
                rendered = content.strip()
        elif isinstance(content, str):
            rendered = content.strip()
        else:
            rendered = ""
        if not rendered:
            continue
        parts.append(rendered)
        size += len(rendered)
        if size >= limit:
            break
    return "\n".join(reversed(parts))[-limit:]


def is_implementation_intent(intent: str) -> bool:
    return "implement_module" in str(intent).casefold()


def implementation_requested(messages: Sequence[Mapping[str, Any]]) -> bool:
    return is_implementation_intent(structured_user_intent(messages))


__all__ = [
    "implementation_requested",
    "is_implementation_intent",
    "structured_user_intent",
]
