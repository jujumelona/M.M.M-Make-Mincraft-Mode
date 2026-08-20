from __future__ import annotations

"""Canonical permission identity for narrow model-facing tool interfaces.

A model-facing ACI may expose a safer/narrower argument shape than the underlying
first-party MCP tool, but it must not create a second authorization namespace.  The
canonical tool remains the sole owner of Skill, role and stage permission.
"""

_TOOL_PERMISSION_ALIASES = {
    "apply_source_edit": "apply_source_patch",
}


def canonical_model_tool(name: str) -> str:
    value = str(name).strip()
    return _TOOL_PERMISSION_ALIASES.get(value, value)


def is_model_tool_alias(name: str) -> bool:
    return str(name).strip() in _TOOL_PERMISSION_ALIASES


__all__ = ["canonical_model_tool", "is_model_tool_alias"]
