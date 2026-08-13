from __future__ import annotations

import sys
from functools import wraps
from typing import Any, Mapping, Sequence

_CORE = ("inspect_existing_mod", "search_project_rag", "search_code_rag")
_EXTERNAL = ("external_mcp_capabilities", "external_mcp_schema", "external_mcp_call")


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def install(max_agent_owner: Any) -> None:
    module = max_agent_owner
    if not hasattr(module, "select_tool_schemas"):
        module = sys.modules[str(getattr(max_agent_owner, "__module__", ""))]
    current = module.select_tool_schemas
    if getattr(current, "_mmm_required_tool_guard", False):
        return

    @wraps(current)
    def guarded(
        router: Any,
        *,
        role: str,
        query: str,
        tool_schemas: Sequence[Mapping[str, Any]],
        require_fresh_evidence: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        tools = tuple(tool_schemas)
        ranked = list(
            current(
                router,
                role=role,
                query=query,
                tool_schemas=tools,
                require_fresh_evidence=require_fresh_evidence,
            )
        )
        available = {_name(schema): schema for schema in tools if _name(schema)}
        required: list[str] = []
        if role in {"coder", "coder_safe"}:
            required.extend(name for name in _CORE if name in available)
        if any(name in available for name in _EXTERNAL):
            required.extend(
                name for name in _EXTERNAL if name in available and name not in required
            )
        limit = max(5, min(8, len(required)))
        required_set = set(required)
        chosen = [available[name] for name in required]
        chosen_names = set(required)
        for schema in ranked:
            name = _name(schema)
            if not name or name in chosen_names or name in required_set:
                continue
            chosen.append(schema)
            chosen_names.add(name)
            if len(chosen) >= limit:
                break
        order = {_name(schema): index for index, schema in enumerate(tools)}
        chosen.sort(key=lambda schema: order.get(_name(schema), len(order)))
        return tuple(chosen)

    guarded._mmm_required_tool_guard = True  # type: ignore[attr-defined]
    guarded.__wrapped__ = current  # type: ignore[attr-defined]
    module.select_tool_schemas = guarded


__all__ = ["install"]
