from __future__ import annotations

"""Per-turn causal tool exposure for the live retrieve/act/observe loop."""

import threading
from contextvars import ContextVar
from typing import Any, Mapping, Sequence

from .causal_tool_graph import executable_frontier, verified_state_from_messages
from .grounding_policy import host_baseline_causal_facts

_AUTHORIZED_TOOLS: ContextVar[tuple[Mapping[str, Any], ...]] = ContextVar(
    "mmm_causal_authorized_tools", default=()
)
_AUTHORIZED_PREFERENCE: ContextVar[tuple[tuple[str, int], ...]] = ContextVar(
    "mmm_causal_authorized_preference", default=()
)
_CURRENT_FRONTIER_NAMES: ContextVar[tuple[str, ...] | None] = ContextVar(
    "mmm_causal_current_frontier_names", default=None
)
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"


class FrontierExecutionGate:
    """Thread-safe execution boundary shared by the adapter and runtime proxy.

    ContextVars are useful for request-local planning metadata but are not inherited by
    ``ThreadPoolExecutor`` worker threads. The model router intentionally executes
    independent read tools in those workers, so the actual host execution boundary
    must live in a shared object rather than relying on thread-local context state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._visible: tuple[str, ...] | None = None

    def set_visible(self, names: Sequence[str]) -> None:
        with self._lock:
            self._visible = tuple(str(name) for name in names)

    def visible_names(self) -> tuple[str, ...] | None:
        with self._lock:
            return self._visible

    def clear(self) -> None:
        with self._lock:
            self._visible = None


def remember_authorized_tools(
    tools: Sequence[Mapping[str, Any]],
    preference: Mapping[str, int] | None = None,
) -> None:
    """Remember the security-filtered surface and query-specific tie-break order."""

    _AUTHORIZED_TOOLS.set(tuple(tools))
    if preference is not None:
        _AUTHORIZED_PREFERENCE.set(
            tuple(
                sorted(
                    ((str(name), int(rank)) for name, rank in preference.items()),
                    key=lambda item: item[1],
                )
            )
        )


def authorized_tools(fallback: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    value = _AUTHORIZED_TOOLS.get()
    return value or tuple(fallback)


def authorized_tool_preference() -> dict[str, int]:
    return dict(_AUTHORIZED_PREFERENCE.get())


def current_frontier_names() -> tuple[str, ...] | None:
    """Return the exact schemas shown on the most recent model turn in this context."""

    return _CURRENT_FRONTIER_NAMES.get()


def clear_current_frontier() -> None:
    _CURRENT_FRONTIER_NAMES.set(None)


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _forced_tool_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    if str(tool_choice.get("type", "")).strip() != "function":
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


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
        execution_gate: FrontierExecutionGate | None = None,
    ) -> None:
        self.inner = inner
        self.stage = stage
        self.role = role
        self.require_fresh_evidence = require_fresh_evidence
        self.frontier_limit = max(1, min(int(frontier_limit), 3))
        self.execution_gate = execution_gate

    def _publish_frontier(self, names: Sequence[str]) -> None:
        normalized = tuple(str(name) for name in names)
        _CURRENT_FRONTIER_NAMES.set(normalized)
        if self.execution_gate is not None:
            self.execution_gate.set_visible(normalized)

    def generate_turn(self, request: Any) -> Any:
        from .causal_tool_frontier_contract import goals_for_query
        from .model_adapters import GenerationRequest, ModelConfigurationError

        # The core loop intentionally emits an explicit tools=() request for
        # fixed-point/final synthesis. That is a control signal, not a new planning
        # round; never resurrect the broader authorization ContextVar on that turn.
        if not request.tools and request.tool_choice is None:
            self._publish_frontier(())
            return self.inner.generate_turn(request)

        candidates = authorized_tools(request.tools)
        if not candidates:
            self._publish_frontier(())
            return self.inner.generate_turn(request)
        by_name = {_name(schema): schema for schema in candidates if _name(schema)}
        forced_name = _forced_tool_name(request.tool_choice)
        if forced_name:
            forced_schema = by_name.get(forced_name)
            if forced_schema is None:
                raise ModelConfigurationError(
                    f"Host-forced tool {forced_name!r} is outside the authorized causal frontier surface."
                )
            state: frozenset[str] = frozenset()
            goals: tuple[str, ...] = ()
            names = (forced_name,)
            selected = (forced_schema,)
            tool_choice = request.tool_choice
            parallel_tool_calls = request.parallel_tool_calls
        else:
            state_facts = set(
                verified_state_from_messages(
                    request.messages,
                    candidates,
                    require_fresh_evidence=self.require_fresh_evidence,
                )
            )
            state_facts.update(host_baseline_causal_facts(request.messages))
            state = frozenset(state_facts)
            query = _query(request.messages)
            goals = tuple(goals_for_query(query))
            names = executable_frontier(
                candidates,
                state=state,
                goals=goals,
                limit=self.frontier_limit,
                max_depth=8,
                preference=authorized_tool_preference(),
            )
            selected = tuple(by_name[name] for name in names if name in by_name)
            tool_choice = "auto" if selected else None
            parallel_tool_calls = True if selected else False
        self._publish_frontier(tuple(_name(schema) for schema in selected))

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
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            task=getattr(request, "task", ""),
            prompt=getattr(request, "prompt", ""),
            metadata=getattr(request, "metadata", {}),
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
    "FrontierExecutionGate",
    "authorized_tool_preference",
    "authorized_tools",
    "clear_current_frontier",
    "current_frontier_names",
    "remember_authorized_tools",
]