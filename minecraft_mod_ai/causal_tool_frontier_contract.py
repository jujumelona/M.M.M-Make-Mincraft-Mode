from __future__ import annotations

"""Expose only host-authorized tools on a verified causal next-action frontier."""

import os
import sys
from functools import wraps
from typing import Any, Mapping, Sequence

from .causal_tool_graph import infer_verified_state, shortest_causal_frontier

_CODER_CORE = ("inspect_existing_mod", "search_project_rag", "search_code_rag")
_EXTERNAL = ("external_mcp_capabilities", "external_mcp_schema", "external_mcp_call")


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _goals(query: str) -> tuple[str, ...]:
    value = query.casefold()
    goals: list[str] = []
    marker_groups = (
        ("external", ("external mcp", "mcp server", "capability", "외부 mcp")),
        ("verify", ("repair", "error", "fail", "compile", "diagnostic", "verify", "test", "검증", "오류", "실패", "수리")),
        ("runtime", ("runtime", "playtest", "server", "client", "screenshot", "런타임", "플레이테스트")),
        ("evidence", ("api", "version", "mapping", "yarn", "research", "evidence", "검색", "근거", "버전")),
        ("observe", ("existing", "project", "inspect", "current", "기존", "프로젝트", "확인")),
        ("act", ("generate", "create", "patch", "modify", "fix", "write", "만들", "수정", "고쳐")),
    )
    for goal, markers in marker_groups:
        if any(marker in value for marker in markers):
            goals.append(goal)
    return tuple(goals or ("observe",))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def install(max_agent_owner: Any) -> None:
    module = max_agent_owner
    if not hasattr(module, "select_tool_schemas"):
        module = sys.modules[str(getattr(max_agent_owner, "__module__", ""))]
    current = module.select_tool_schemas
    if getattr(current, "_mmm_causal_tool_frontier", False):
        return

    @wraps(current)
    def causal_frontier(
        router: Any,
        *,
        role: str,
        query: str,
        tool_schemas: Sequence[Mapping[str, Any]],
        require_fresh_evidence: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        available = tuple(tool_schemas)
        ranked = tuple(
            current(
                router,
                role=role,
                query=query,
                tool_schemas=available,
                require_fresh_evidence=require_fresh_evidence,
            )
        )
        if len(ranked) <= 2:
            return ranked

        names = {_name(schema) for schema in available}
        protected: list[str] = []
        if role in {"coder", "coder_safe"}:
            protected.extend(name for name in _CODER_CORE if name in names)
        if any(name in names for name in _EXTERNAL):
            protected.extend(name for name in _EXTERNAL if name in names)

        state = infer_verified_state(
            query=query,
            tool_schemas=available,
            require_fresh_evidence=require_fresh_evidence,
        )
        goals = _goals(query)
        path = shortest_causal_frontier(
            available,
            state=state,
            goals=goals,
            protected=protected,
            max_depth=_env_int("MMM_CAUSAL_TOOL_MAX_DEPTH", 4, minimum=1, maximum=8),
        )

        # If no causal path can prove progress, keep the protected host surface plus
        # the single highest-ranked optional tool rather than widening to all tools.
        selected_names = list(path)
        if len(selected_names) <= len(protected):
            for schema in ranked:
                name = _name(schema)
                if name and name not in selected_names:
                    selected_names.append(name)
                    break

        max_optional = _env_int("MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=1, maximum=5)
        hard = set(protected)
        optional = [name for name in selected_names if name not in hard]
        selected_names = [*protected, *optional[:max_optional]]

        by_name = {_name(schema): schema for schema in available if _name(schema)}
        order = {_name(schema): index for index, schema in enumerate(available)}
        result = tuple(
            sorted(
                (by_name[name] for name in selected_names if name in by_name),
                key=lambda schema: order.get(_name(schema), len(order)),
            )
        )
        print(
            "causal tool frontier:",
            f"role={role}",
            f"state={','.join(sorted(state))}",
            f"goals={','.join(goals)}",
            f"path={','.join(path)}",
            f"exposed={len(result)}",
            flush=True,
        )
        return result

    causal_frontier._mmm_causal_tool_frontier = True  # type: ignore[attr-defined]
    causal_frontier.__wrapped__ = current  # type: ignore[attr-defined]
    module.select_tool_schemas = causal_frontier


__all__ = ["install"]
