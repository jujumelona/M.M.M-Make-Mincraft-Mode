from __future__ import annotations

"""Canonical permission identity for narrow model-facing tool interfaces.

A model-facing ACI may expose a safer/narrower argument shape than the underlying
first-party MCP tool, but it must not create a second authorization namespace.  The
canonical tool remains the sole owner of Skill, role and stage permission.
"""

from collections.abc import Iterable
from types import MappingProxyType

_TOOL_PERMISSION_ALIASES = MappingProxyType(
    {
        "apply_source_edit": "apply_source_patch",
        "patch_file": "apply_source_patch",
    }
)


def canonical_model_tool(name: str) -> str:
    value = str(name).strip()
    return _TOOL_PERMISSION_ALIASES.get(value, value)


def is_model_tool_alias(name: str) -> bool:
    return str(name).strip() in _TOOL_PERMISSION_ALIASES


def resolve_exposed_model_tool(
    emitted_name: str,
    exposed_names: Iterable[str],
) -> str | None:
    """Resolve one emitted name through canonical identity to an actually exposed tool."""

    emitted = str(emitted_name).strip()
    exposed = tuple(
        dict.fromkeys(
            value
            for raw in exposed_names
            if (value := str(raw).strip())
        )
    )
    if emitted in exposed:
        return emitted
    emitted_canonical = canonical_model_tool(emitted)
    matches = tuple(
        candidate
        for candidate in exposed
        if canonical_model_tool(candidate) == emitted_canonical
    )
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "canonical_model_tool",
    "is_model_tool_alias",
    "resolve_exposed_model_tool",
]
