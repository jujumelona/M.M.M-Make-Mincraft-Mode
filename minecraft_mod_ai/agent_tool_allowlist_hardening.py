from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

_MARKER = "__mmm_preserve_generation_tool_allowlist_v1__"


def harden_agent_tool_allowlist() -> None:
    """Preserve reviewed stage tools when late contracts append dynamic schemas.

    Dynamic generation contracts may append one host implementation tool, but they
    must never replace the allowlist already derived from first-party MCP schemas.
    This wrapper restores monotonic union semantics: only names that were already
    reviewed or are present in the returned schema set remain executable.
    """
    from .agent_tool_runtime import AgentToolRuntime

    current = AgentToolRuntime.tool_schemas
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def tool_schemas(self: Any, stage: str):
        selected = self._stage(stage)
        with self._lock:
            before = frozenset(self._allowed_tool_cache.get(selected, frozenset()))
        schemas = current(self, selected)
        returned = frozenset(
            name
            for schema in schemas
            if isinstance(schema, Mapping)
            and (name := _schema_name(schema))
        )
        with self._lock:
            after = frozenset(self._allowed_tool_cache.get(selected, frozenset()))
            self._allowed_tool_cache[selected] = before | after | returned
        return schemas

    setattr(tool_schemas, _MARKER, True)
    AgentToolRuntime.tool_schemas = tool_schemas


def _schema_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


__all__ = ["harden_agent_tool_allowlist"]
