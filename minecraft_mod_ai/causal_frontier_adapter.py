from __future__ import annotations

"""Per-turn causal tool exposure for the live retrieve/act/observe loop."""

from contextvars import ContextVar
from typing import Any, Mapping, Sequence

from .causal_tool_graph import executable_frontier, verified_state_from_messages

_AUTHORIZED_TOOLS: ContextVar[tuple[Mapping[str, Any], ...]] = ContextVar(
    "mmm_causal_authorized_tools", default=()
)
_AUTHORIZED_PREFERENCE: ContextVar[tuple[tuple[str, int], ...]] = ContextVar(
    "mmm_causal_authorized_preference", default=()
)
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"


def remember_authorized_tools(
    tools: Sequence[Mapping[str, Any]],
    preference: Mapping[str, int] | None = None,
) -> None:
    """Remember the security-filtered surface and query-specific tie-break order."""

    _AUTHORIZED_TOOLS.set(tuple(tools))
    if preference is not None:
        _AUTHORIZED_PREFERENCE.set(
            tuple(sorted(((str(name), int(rank)) for name, rank in preference.items()), key=lambda item: item[1]))
        )


def authorized_tools(fallback: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    value = _AUTHORIZED_TOOLS.get()
    return value or tuple(fallback)


def authorized_tool_preference() -> dict[str, int]:
    return dict(_AUTHORIZED_PREFERENCE.get())


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _query(messages: Sequence[Mapping[str, Any]]) -> str:
    """Recover terminal intent from user turns only."""

    parts: list[str] = []
    for message in reversed(messages):
        if str(message.get("role", "")).casefold() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        if sum(len(item) for item in parts) >= 12_000:
            break
    return "\n".join(reversed(parts))[-12_000:]


def _with_capability_context(
    messages: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    role: str,
    tools: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    from .agent_capability_context import build_agent_capability_context

    copied = [
        dict(message)
        for message in messages
        if not (
            str(message.get("role", "")) == "system"
            and isinstance(message.get("content"), str)
            and str(message.get("content", "")).startswith(_CAPABILITY_PREFIX)
        )
    ]
    if not tools:
        return tuple(copied)
    insert_at = 0
    while insert_at < len(copied) and str(copied[insert_at].get("role", "")) == "system":
        insert_at += 1
    copied.insert(
        insert_at,
        {
            "role": "system",
            "content": build_agent_capability_context(stage, tools, model_role=role),
        },
    )
    return tuple(copied)


class CausalFrontierAdapter:
    """Delegate adapter that recalculates the next executable edge every turn."""

    def __init__(
        self,
        inner: Any,
        *,
        stage: str,
        role: str,
        require_fresh_evidence: bool,
        frontier_limit: int = 3,
    ) -> None:
        self.inner = inner
        self.stage = stage
        self.role = role
        self.require_fresh_evidence = require_fresh_evidence
        self.frontier_limit = max(1, min(int(frontier_limit), 3))

    def generate_turn(self, request: Any) -> Any:
        from .causal_tool_frontier_contract import goals_for_query
        from .model_adapters import GenerationRequest

        candidates = authorized_tools(request.tools)
        if not candidates:
            return self.inner.generate_turn(request)
        state = verified_state_from_messages(
            request.messages,
            candidates,
            require_fresh_evidence=self.require_fresh_evidence,
        )
        query = _query(request.messages)
        goals = goals_for_query(query)
        names = executable_frontier(
            candidates,
            state=state,
            goals=goals,
            limit=self.frontier_limit,
            max_depth=8,
            preference=authorized_tool_preference(),
        )
        by_name = {_name(schema): schema for schema in candidates if _name(schema)}
        selected = tuple(by_name[name] for name in names if name in by_name)

        rebuilt = GenerationRequest(
            messages=_with_capability_context(
                request.messages,
                stage=self.stage,
                role=self.role,
                tools=selected,
            ),
            media_paths=request.media_paths,
            response_format=request.response_format,
            response_schema=request.response_schema,
            tools=selected,
            tool_choice="auto" if selected else None,
            parallel_tool_calls=True if selected else False,
        )
        print(
            "causal per-turn frontier:",
            f"role={self.role}",
            f"state={','.join(sorted(state))}",
            f"goals={','.join(goals)}",
            f"tools={','.join(names)}",
            flush=True,
        )
        return self.inner.generate_turn(rebuilt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


__all__ = [
    "CausalFrontierAdapter",
    "authorized_tool_preference",
    "authorized_tools",
    "remember_authorized_tools",
]
