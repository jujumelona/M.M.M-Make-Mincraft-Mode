from __future__ import annotations

"""Per-turn causal tool exposure for the live retrieve/act/observe loop."""

import json
import threading
from contextvars import ContextVar
from dataclasses import replace
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
_SOURCE_MUTATION_NAMES = frozenset(
    {
        "apply_source_patch",
        "apply_source_edit",
        "apply_java_operations",
        "repair_project",
    }
)
_MUTATION_FAILURE_STATUSES = frozenset(
    {
        "FAIL",
        "FAILED",
        "ERROR",
        "UNAVAILABLE",
        "PARTIAL",
        "BLOCKED",
        "INVALID",
        "REJECTED",
        "CANCELLED",
        "CANCELED",
        "TIMEOUT",
    }
)
_STALE_EVIDENCE_FACTS = frozenset(
    {
        "code_evidence",
        "project_evidence",
        "ecosystem_evidence",
        "external_observation",
        "evidence_ready",
    }
)


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


def _tool_payload(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _has_semantic_failure(payload: Mapping[str, Any]) -> bool:
    root = payload.get("result", payload)
    if not isinstance(root, Mapping):
        return False
    results: list[Mapping[str, Any]] = [root]
    for key in ("structured_content", "structured", "parsed_text"):
        nested = root.get(key)
        if isinstance(nested, Mapping):
            results.append(nested)
    return any(
        str(result.get("status", "")).strip().upper() in _MUTATION_FAILURE_STATUSES
        for result in results
    )


def _latest_failed_source_mutation(
    messages: Sequence[Mapping[str, Any]],
) -> int | None:
    """Return the most recent failed source-mutation observation in the transcript."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role", "")).strip().casefold() != "tool":
            continue
        if str(message.get("name", "")).strip() not in _SOURCE_MUTATION_NAMES:
            continue
        payload = _tool_payload(message)
        if payload is None:
            continue
        if payload.get("ok") is not True or _has_semantic_failure(payload):
            return index
    return None


def _failed_mutation_needs_evidence_refresh(
    messages: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> bool:
    """Invalidate only evidence older than the latest failed source mutation.

    A failed edit means the evidence that justified that exact mutation did not produce
    a valid host-side change. Reusing it pins the planner on the same mutation-only
    frontier. Evidence successfully observed after the failure starts a new repair
    epoch and makes mutation legal again.
    """

    failed_at = _latest_failed_source_mutation(messages)
    if failed_at is None:
        return False
    refreshed = verified_state_from_messages(
        messages[failed_at + 1 :],
        candidates,
        require_fresh_evidence=True,
    )
    return "evidence_ready" not in refreshed


def _structured_intent(payload: Mapping[str, Any]) -> str:
    """Extract routing intent without hauling evidence/context blobs into the query."""

    parts: list[str] = []
    for key in _INTENT_FIELDS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()[:4096]}")
    rules = payload.get("rules")
    if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes, bytearray)):
        for value in rules[:16]:
            if isinstance(value, str) and value.strip():
                parts.append(f"rule: {value.strip()[:1024]}")
    module = payload.get("module")
    if isinstance(module, Mapping):
        for key in ("module_id", "kind", "name", "type"):
            value = module.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(f"module.{key}: {value.strip()[:512]}")
    return "\n".join(parts)


def _intent_text(content: Any) -> str:
    if isinstance(content, Mapping):
        extracted = _structured_intent(content)
        return extracted or json.dumps(content, ensure_ascii=False, default=str)
    if not isinstance(content, str) or not content.strip():
        return ""
    raw = content.strip()
    if not raw.startswith("{"):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, Mapping):
        return raw
    extracted = _structured_intent(payload)
    return extracted or raw


def _query(messages: Sequence[Mapping[str, Any]]) -> str:
    """Recover bounded terminal intent from user turns only.

    Structured user requests can contain tens of kilobytes of repository evidence.
    Routing from the last 12 KiB of that blob can discard the leading ``phase`` and
    ``task`` fields entirely. Extract explicit intent fields before applying the byte
    bound so evidence payload size cannot change the terminal causal goal.
    """

    parts: list[str] = []
    for message in reversed(messages):
        if str(message.get("role", "")).casefold() != "user":
            continue
        value = _intent_text(message.get("content"))
        if value:
            parts.append(value)
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
        authorized_surface: Sequence[Mapping[str, Any]] = (),
        preference: Mapping[str, int] | None = None,
    ) -> None:
        self.inner = inner
        self.stage = stage
        self.role = role
        self.require_fresh_evidence = require_fresh_evidence
        self.frontier_limit = max(1, min(int(frontier_limit), 3))
        self.execution_gate = execution_gate
        # Freeze these once for the whole live tool loop. Nested model/retrieval calls
        # may update the compatibility ContextVars, but they must never replace this
        # coder turn's security-filtered authorization or query preference.
        self.authorized_surface = tuple(authorized_surface)
        self.preference = dict(preference or {})
        self._last_stale_frontier: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def _publish_frontier(self, names: Sequence[str]) -> None:
        normalized = tuple(str(name) for name in names)
        _CURRENT_FRONTIER_NAMES.set(normalized)
        if self.execution_gate is not None:
            self.execution_gate.set_visible(normalized)

    def _reset_stale_guard(self) -> None:
        self._last_stale_frontier = None

    def generate_turn(self, request: Any) -> Any:
        from .causal_tool_frontier_contract import goals_for_query
        from .model_adapters import ModelConfigurationError

        # The core loop intentionally emits an explicit tools=() request for
        # fixed-point/final synthesis. That is a control signal, not a new planning
        # round; never resurrect the broader authorization ContextVar on that turn.
        if not request.tools and request.tool_choice is None:
            self._publish_frontier(())
            self._reset_stale_guard()
            return self.inner.generate_turn(request)

        candidates = self.authorized_surface or authorized_tools(request.tools)
        if not candidates:
            self._publish_frontier(())
            self._reset_stale_guard()
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
            if _failed_mutation_needs_evidence_refresh(request.messages, candidates):
                state_facts.difference_update(_STALE_EVIDENCE_FACTS)
            state = frozenset(state_facts)
            query = _query(request.messages)
            goals = tuple(goals_for_query(query))
            names = executable_frontier(
                candidates,
                state=state,
                goals=goals,
                limit=self.frontier_limit,
                max_depth=8,
                preference=self.preference or authorized_tool_preference(),
            )
            selected = tuple(by_name[name] for name in names if name in by_name)
            tool_choice = "auto" if selected else None
            parallel_tool_calls = True if selected else False
        selected_names = tuple(_name(schema) for schema in selected)
        self._publish_frontier(selected_names)

        # GenerationRequest is a frozen dataclass. ``replace`` preserves every field
        # owned by upstream contracts (including future additions) while changing only
        # the per-turn causal surface and injected capability context. The broader
        # validation surface lets a transport parser understand a stale but authorized
        # tool reference; FrontierExecutionGate still rejects execution unless that
        # tool is present in ``selected`` on this exact turn.
        rebuilt = replace(
            request,
            messages=_with_capability_context(
                request.messages,
                stage=self.stage,
                role=self.role,
                tools=selected,
            ),
            tools=selected,
            tool_validation_schemas=candidates,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        print(
            "causal per-turn frontier:",
            f"role={self.role}",
            f"state={','.join(sorted(state))}",
            f"goals={','.join(goals)}",
            f"tools={','.join(names)}",
            flush=True,
        )
        turn = self.inner.generate_turn(rebuilt)
        stale_names = tuple(
            sorted(
                {
                    str(call.name)
                    for call in getattr(turn, "tool_calls", ())
                    if str(call.name) in by_name and str(call.name) not in selected_names
                }
            )
        )
        if not stale_names:
            self._reset_stale_guard()
            return turn

        fingerprint = (selected_names, stale_names)
        if fingerprint == self._last_stale_frontier:
            raise ModelConfigurationError(
                "Model repeated stale authorized tool calls without causal frontier "
                f"progress: stale={','.join(stale_names)} "
                f"visible={','.join(selected_names)}"
            )
        self._last_stale_frontier = fingerprint
        return turn

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
