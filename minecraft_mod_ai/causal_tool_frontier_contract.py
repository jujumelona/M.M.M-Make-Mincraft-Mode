from __future__ import annotations

"""Reduce relevance-ranked tools to the smallest useful next-action frontier."""

import os
import sys
from functools import wraps
from typing import Any, Mapping, Sequence

_EVIDENCE = ("search_code_rag", "search_project_rag")
_INSPECT = ("inspect_existing_mod",)
_EXTERNAL = ("external_mcp_capabilities", "external_mcp_schema", "external_mcp_call")


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _document(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    if not isinstance(fn, Mapping):
        return ""
    return (str(fn.get("name", "")) + " " + str(fn.get("description", ""))).casefold()


def _effect(schema: Mapping[str, Any]) -> str:
    text = _document(schema)
    name = _name(schema).casefold()
    if name in _EXTERNAL or "external mcp" in text:
        return "external"
    if any(token in name for token in ("search", "discover")):
        return "evidence"
    if any(token in name for token in ("inspect", "read", "status", "logs", "symbols")):
        return "observe"
    if any(token in name for token in ("diagnostic", "validate", "quality", "test", "smoke")):
        return "verify"
    if any(token in name for token in ("patch", "generate", "write", "apply", "execute", "command")):
        return "act"
    if any(token in name for token in ("runtime", "mineflayer", "blockbench")):
        return "runtime"
    return "other"


def _goals(query: str) -> tuple[str, ...]:
    value = query.casefold()
    goals: list[str] = []
    marker_groups = (
        ("external", ("external mcp", "mcp server", "capability", "외부 mcp")),
        ("verify", ("error", "fail", "compile", "diagnostic", "verify", "test", "검증", "오류", "실패")),
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

        goals = _goals(query)
        ambiguous = len(set(goals)) >= 3
        width = _env_int("MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=2, maximum=5)
        if ambiguous:
            width = min(5, max(width, 4))
        by_name = {_name(schema): schema for schema in available if _name(schema)}
        selected: list[Mapping[str, Any]] = []
        selected_names: set[str] = set()

        def add(name: str) -> None:
            schema = by_name.get(name)
            if schema is not None and name not in selected_names:
                selected.append(schema)
                selected_names.add(name)

        # Fresh production code needs evidence before an action. Keep both evidence
        # planes; inspect-existing is causal only for an existing/repair context.
        if require_fresh_evidence and role in {"coder", "coder_safe"}:
            for name in _EVIDENCE:
                add(name)
            if "observe" in goals or "verify" in goals:
                for name in _INSPECT:
                    add(name)

        external_needed = "external" in goals or any(_name(item) in _EXTERNAL for item in ranked[:2])
        if external_needed:
            # External MCP discovery/call is an atomic capability/schema/call chain.
            for name in _EXTERNAL:
                add(name)

        preference = {
            "verify": ("verify", "observe", "evidence", "act", "runtime"),
            "runtime": ("runtime", "verify", "observe", "evidence", "act"),
            "evidence": ("evidence", "observe", "verify", "act", "runtime"),
            "observe": ("observe", "evidence", "verify", "act", "runtime"),
            "act": ("evidence", "observe", "act", "verify", "runtime"),
            "external": ("external", "evidence", "observe", "verify", "act"),
        }
        ordered_effects: list[str] = []
        for goal in goals:
            for effect in preference.get(goal, (goal,)):
                if effect not in ordered_effects:
                    ordered_effects.append(effect)

        ranked_index = {_name(schema): index for index, schema in enumerate(ranked)}
        candidates = sorted(
            ranked,
            key=lambda schema: (
                ordered_effects.index(_effect(schema)) if _effect(schema) in ordered_effects else len(ordered_effects),
                ranked_index.get(_name(schema), len(ranked)),
            ),
        )
        target_width = max(width, len(selected))
        for schema in candidates:
            name = _name(schema)
            if not name or name in selected_names:
                continue
            selected.append(schema)
            selected_names.add(name)
            if len(selected) >= target_width:
                break

        # Preserve original stage order for stable schema prefixes/KV reuse.
        order = {_name(schema): index for index, schema in enumerate(available)}
        selected.sort(key=lambda schema: order.get(_name(schema), len(order)))
        result = tuple(selected[:target_width])
        print(
            "causal tool frontier:",
            f"role={role}",
            f"goals={','.join(goals)}",
            f"ranked={len(ranked)}",
            f"exposed={len(result)}",
            flush=True,
        )
        return result

    causal_frontier._mmm_causal_tool_frontier = True  # type: ignore[attr-defined]
    causal_frontier.__wrapped__ = current  # type: ignore[attr-defined]
    module.select_tool_schemas = causal_frontier


__all__ = ["install"]
