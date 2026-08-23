from __future__ import annotations

"""Keep writable coder tool routes intact across late runtime composition.

The small-model selector may expose only the first causal action, while the live
causal adapter must retain the complete security-filtered tool surface so later
observations can unlock source mutation. This contract is deliberately installed
after the progress-aware retrieval loop: it composes both policies instead of
allowing either late wrapper to replace the other.
"""

import json
from collections import Counter
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from .causal_frontier_adapter import (
    CausalFrontierAdapter,
    FrontierExecutionGate,
    authorized_tool_preference,
    authorized_tools,
    clear_current_frontier,
)
from .causal_tool_frontier_contract import _FrontierRuntimeProxy
from .causal_tool_graph import shortest_causal_path
from .source_mutation_contract import mutation_observation_applied

_MARKER = "_mmm_coder_tool_route_integrity_v1"
_QUERY_MARKER = "_mmm_user_only_tool_routing_query_v1"
_GOAL_MARKER = "_mmm_implementation_goal_priority_v1"
_MUTATION_PRIORITY = (
    "apply_source_patch",
    "apply_source_edit",
    "apply_java_operations",
    "repair_project",
)
_MUTATION_TOOLS = frozenset(_MUTATION_PRIORITY)
_MUTATION_ROUTE_FAILURE_LIMIT = 2


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


def _named_tool_choice_name(tool_choice: Any) -> str:
    """Return one host-forced function name without weakening other choice modes."""

    if not isinstance(tool_choice, Mapping):
        return ""
    if str(tool_choice.get("type", "")).strip() != "function":
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _user_only_request_query(messages: Sequence[Mapping[str, Any]]) -> str:
    """Route from user intent, never from injected capability boilerplate."""

    parts: list[str] = []
    for message in reversed(messages):
        if str(message.get("role", "")).strip().casefold() != "user":
            continue
        value = message.get("content")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        if sum(len(item) for item in parts) >= 12_000:
            break
    return "\n".join(reversed(parts))[-12_000:]


def _is_implementation_request(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Recognize the host-owned custom-module write phase without guessing from prose."""

    for message in reversed(messages):
        if str(message.get("role", "")).strip().casefold() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        lowered = content.casefold()
        if "implement_module" in lowered:
            return True
        if not content.lstrip().startswith("{"):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("phase", "")).strip().casefold() == "implement_module":
            return True
    return False


def _source_mutation_applied(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Consume the canonical host mutation proof instead of transport success."""

    return any(mutation_observation_applied(message) for message in reversed(messages))


def _mutation_failure_counts(messages: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Count explicit failed mutation observations already visible to this turn."""

    counts: Counter[str] = Counter()
    for message in messages:
        if str(message.get("role", "")).strip().casefold() != "tool":
            continue
        name = str(message.get("name", "")).strip()
        if name not in _MUTATION_TOOLS:
            continue
        content = message.get("content")
        payload: Any = content
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
        if isinstance(payload, Mapping) and payload.get("ok") is False:
            counts[name] += 1
    return counts


def _preferred_visible_mutation(
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Choose the cheapest visible mutation route that has not locally exhausted."""

    visible = {_tool_name(schema) for schema in tools}
    failures = _mutation_failure_counts(messages)
    for name in _MUTATION_PRIORITY:
        if name in visible and failures[name] < _MUTATION_ROUTE_FAILURE_LIMIT:
            return name
    return ""


def _force_tool_choice(request: Any, tool_name: str) -> Any:
    selected = tuple(
        schema
        for schema in request.tools
        if _tool_name(schema) == tool_name
    )
    if not selected:
        return request
    remainder = tuple(
        schema
        for schema in request.tools
        if _tool_name(schema) != tool_name
    )
    return replace(
        request,
        tools=selected + remainder,
        tool_choice={"type": "function", "function": {"name": tool_name}},
        parallel_tool_calls=False,
    )


def _require_mutation_surface(
    tools: Sequence[Mapping[str, Any]],
    *,
    messages: Sequence[Mapping[str, Any]],
    stage: str,
    role: str,
) -> None:
    if stage != "generation" or role not in {"coder", "coder_safe"}:
        return
    if not _is_implementation_request(messages):
        return

    from .model_adapters import ModelConfigurationError

    names = {_tool_name(schema) for schema in tools}
    if not names & _MUTATION_TOOLS:
        raise ModelConfigurationError(
            "Writable coder generation has no authorized source-mutation tool. "
            "The host refuses to run a model turn that cannot produce a source diff."
        )

    route = shortest_causal_path(
        tools,
        state=frozenset({"workspace_bound"}),
        goals=("repair",),
        max_depth=8,
    )
    if not route or not any(name in _MUTATION_TOOLS for name in route):
        raise ModelConfigurationError(
            "Writable coder generation has source-mutation tools but no reachable "
            "causal route from workspace observation/evidence to a source edit."
        )


class _WritableProgressAdapter:
    """Keep an implementation turn alive until a reviewed source mutation succeeds.

    Retrieval remains model-owned. Once mutation is causal/legal, the host prefers the
    cheapest reviewed route. Two explicit failures on that same visible route permit a
    local fail-over to the next reviewed mutation route; exhausting every visible route
    fails closed. This bounded route fail-over does not replay the node or own evidence
    refresh/checkpoints: ``CausalStateLedger`` remains the owner of those transitions.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def generate_turn(self, request: Any) -> Any:
        from .model_adapters import ModelConfigurationError

        if not _is_implementation_request(request.messages):
            return self.inner.generate_turn(request)

        if not request.tools:
            if request.tool_choice is None and not _source_mutation_applied(request.messages):
                raise ModelConfigurationError(
                    "Writable coder attempted final synthesis before any reviewed source "
                    "mutation succeeded; refusing an implementation summary without a source diff."
                )
            return self.inner.generate_turn(request)

        forced_name = _named_tool_choice_name(request.tool_choice)
        if request.tool_choice != "auto" and forced_name not in _MUTATION_TOOLS:
            return self.inner.generate_turn(request)

        if _source_mutation_applied(request.messages):
            return self.inner.generate_turn(request)

        failures = _mutation_failure_counts(request.messages)
        if forced_name and failures[forced_name] >= _MUTATION_ROUTE_FAILURE_LIMIT:
            raise ModelConfigurationError(
                f"Writable coder exhausted bounded mutation retry budget for {forced_name!r}."
            )

        mutation = forced_name or _preferred_visible_mutation(request.tools, request.messages)
        visible_mutations = {
            _tool_name(schema)
            for schema in request.tools
            if _tool_name(schema) in _MUTATION_TOOLS
        }
        if not mutation and visible_mutations:
            raise ModelConfigurationError(
                "Writable coder exhausted bounded mutation retry budget for every visible source-mutation route."
            )
        if mutation:
            forced = request if forced_name == mutation else _force_tool_choice(request, mutation)
            turn = self.inner.generate_turn(forced)
            if not turn.tool_calls or mutation not in {call.name for call in turn.tool_calls}:
                raise ModelConfigurationError(
                    f"Writable coder did not execute required source-mutation action {mutation!r}; "
                    "refusing a prose-only implementation turn after mutation became causal/legal."
                )
            return turn

        turn = self.inner.generate_turn(request)
        if turn.tool_calls:
            return turn
        fallback = next(
            (name for schema in request.tools if (name := _tool_name(schema))),
            "",
        )
        if not fallback:
            raise ModelConfigurationError(
                "Writable coder causal frontier contained no executable named tool."
            )
        forced = _force_tool_choice(request, fallback)
        retry = self.inner.generate_turn(forced)
        if not retry.tool_calls or fallback not in {call.name for call in retry.tool_calls}:
            raise ModelConfigurationError(
                f"Writable coder returned prose twice instead of executing causal action {fallback!r}; "
                "refusing to terminate before source mutation."
            )
        return retry

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _run_with_dynamic_frontier(
    current: Any,
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: Any,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    """Run the final tool loop with the complete authorized surface behind a per-turn frontier."""

    complete_surface = tuple(authorized_tools(request.tools))
    complete_preference = dict(authorized_tool_preference())
    _require_mutation_surface(
        complete_surface,
        messages=request.messages,
        stage=stage,
        role=role,
    )
    host_request = replace(
        request,
        tools=complete_surface,
        tool_validation_schemas=complete_surface,
        tool_choice="auto" if complete_surface else None,
        parallel_tool_calls=True if complete_surface else False,
    )
    execution_gate = FrontierExecutionGate()
    wrapped_adapter = CausalFrontierAdapter(
        _WritableProgressAdapter(adapter),
        stage=stage,
        role=role,
        require_fresh_evidence=bool(
            getattr(router, "_agent_require_fresh_evidence", False)
        ),
        frontier_limit=3,
        execution_gate=execution_gate,
        authorized_surface=complete_surface,
        preference=complete_preference,
        request_template=host_request,
    )
    clear_current_frontier()
    try:
        return current(
            router,
            config=config,
            adapter=wrapped_adapter,
            request=host_request,
            runtime=_FrontierRuntimeProxy(runtime, execution_gate),
            stage=stage,
            role=role,
        )
    finally:
        execution_gate.clear()
        clear_current_frontier()


def _install_user_intent_routing(small_model_module: Any) -> None:
    current = small_model_module._request_query
    if getattr(current, _QUERY_MARKER, False):
        return

    @wraps(current)
    def request_query(messages: Sequence[Mapping[str, Any]]) -> str:
        value = _user_only_request_query(messages)
        return value if value else current(messages)

    setattr(request_query, _QUERY_MARKER, True)
    small_model_module._request_query = request_query


def _install_implementation_goal_priority(causal_module: Any) -> None:
    current = causal_module.goals_for_query
    if getattr(current, _GOAL_MARKER, False):
        return

    @wraps(current)
    def goals_for_query(query: str) -> tuple[str, ...]:
        if "implement_module" in str(query).casefold():
            return ("repair",)
        return tuple(current(query))

    setattr(goals_for_query, _GOAL_MARKER, True)
    causal_module.goals_for_query = goals_for_query


def install(
    *,
    model_router_module: Any,
    small_model_module: Any,
    causal_module: Any,
) -> None:
    """Compose late progress-aware retrieval with the already-reviewed causal frontier."""

    _install_user_intent_routing(small_model_module)
    _install_implementation_goal_priority(causal_module)

    cls = model_router_module.ModelRouter
    current = cls._generate_with_tools
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate_with_route_integrity(
        self: Any,
        *,
        config: Any,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        return _run_with_dynamic_frontier(
            current,
            self,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    setattr(generate_with_route_integrity, _MARKER, True)
    generate_with_route_integrity._mmm_dynamic_causal_frontier = True
    generate_with_route_integrity._mmm_progress_aware_causal_composed = True
    generate_with_route_integrity._mmm_writable_coder_fail_closed = True
    generate_with_route_integrity._mmm_writable_coder_progress_forced = True
    generate_with_route_integrity._mmm_writable_coder_route_reachable = True
    generate_with_route_integrity._mmm_writable_coder_mutation_completion_invariant = True
    cls._generate_with_tools = generate_with_route_integrity


__all__ = [
    "_WritableProgressAdapter",
    "_is_implementation_request",
    "_require_mutation_surface",
    "_run_with_dynamic_frontier",
    "_source_mutation_applied",
    "_user_only_request_query",
    "install",
]
