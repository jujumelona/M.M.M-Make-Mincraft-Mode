from __future__ import annotations

"""Keep writable coder tool routes intact across late runtime composition.

The small-model selector may expose only the first causal action, while the live
causal adapter must retain the complete security-filtered tool surface so later
observations can unlock source mutation.  This contract is deliberately installed
after the progress-aware retrieval loop: it composes both policies instead of
allowing either late wrapper to replace the other.
"""

import json
from functools import wraps
from typing import Any, Mapping, Sequence

from .causal_frontier_adapter import (
    CausalFrontierAdapter,
    FrontierExecutionGate,
    authorized_tools,
    clear_current_frontier,
)
from .causal_tool_frontier_contract import _FrontierRuntimeProxy

_MARKER = "_mmm_coder_tool_route_integrity_v1"
_QUERY_MARKER = "_mmm_user_only_tool_routing_query_v1"
_GOAL_MARKER = "_mmm_implementation_goal_priority_v1"
_MUTATION_TOOLS = frozenset(
    {
        "apply_source_patch",
        "apply_source_edit",
        "apply_java_operations",
        "repair_project",
    }
)


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


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
    names = {_tool_name(schema) for schema in tools}
    if names & _MUTATION_TOOLS:
        return
    from .model_adapters import ModelConfigurationError

    raise ModelConfigurationError(
        "Writable coder generation has no authorized source-mutation tool. "
        "The host refuses to run a model turn that cannot produce a source diff."
    )


def _run_with_dynamic_frontier(
    current: Any,
    router: Any,
    *,
    adapter: Any,
    request: Any,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    """Run the final tool loop with the complete authorized surface behind a per-turn frontier."""

    from .model_adapters import GenerationRequest

    complete_surface = authorized_tools(request.tools)
    _require_mutation_surface(
        complete_surface,
        messages=request.messages,
        stage=stage,
        role=role,
    )
    host_request = GenerationRequest(
        messages=request.messages,
        media_paths=request.media_paths,
        response_format=request.response_format,
        response_schema=request.response_schema,
        tools=complete_surface,
        tool_choice="auto" if complete_surface else None,
        parallel_tool_calls=True if complete_surface else False,
    )
    execution_gate = FrontierExecutionGate()
    wrapped_adapter = CausalFrontierAdapter(
        adapter,
        stage=stage,
        role=role,
        require_fresh_evidence=bool(
            getattr(router, "_agent_require_fresh_evidence", False)
        ),
        frontier_limit=3,
        execution_gate=execution_gate,
    )
    clear_current_frontier()
    try:
        return current(
            router,
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
        # Host custom generation embeds evidence/tool metadata in the user JSON.  The
        # explicit phase is stronger terminal intent than incidental strings such as
        # "external MCP" appearing inside that metadata.
        if "implement_module" in str(query).casefold():
            return ("act",)
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
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        return _run_with_dynamic_frontier(
            current,
            self,
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
    cls._generate_with_tools = generate_with_route_integrity


__all__ = [
    "_is_implementation_request",
    "_require_mutation_surface",
    "_run_with_dynamic_frontier",
    "_user_only_request_query",
    "install",
]
