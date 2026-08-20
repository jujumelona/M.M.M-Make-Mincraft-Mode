from __future__ import annotations

"""Keep writable coder tool routes intact across late runtime composition.

The small-model selector may expose only the first causal action, while the live
causal adapter must retain the complete security-filtered tool surface so later
observations can unlock source mutation. This contract is deliberately installed
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
from .causal_tool_graph import shortest_causal_path

_MARKER = "_mmm_coder_tool_route_integrity_v1"
_QUERY_MARKER = "_mmm_user_only_tool_routing_query_v1"
_GOAL_MARKER = "_mmm_implementation_goal_priority_v1"
_HOST_MUTATION_PROOF_KEY = "_mmm_source_mutation"
_MUTATION_PRIORITY = (
    "apply_source_patch",
    "apply_source_edit",
    "apply_java_operations",
    "repair_project",
)
_MUTATION_TOOLS = frozenset(_MUTATION_PRIORITY)
_MUTATION_FAILURE_LIMIT = 2
_FAILURE_STATUSES = frozenset(
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


def _walk_mappings(value: Any):
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            yield current
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            pending.extend(current)


def _has_applied_patch_receipt(payload: Mapping[str, Any]) -> bool:
    """Require the transaction receipt that is emitted only after a real source diff."""

    for item in _walk_mappings(payload):
        if str(item.get("schema_version", "")) != "mmm/source-patch-receipt-v1":
            continue
        if str(item.get("status", "")).strip().upper() != "APPLIED":
            continue
        operations = item.get("operations")
        if isinstance(operations, Sequence) and not isinstance(
            operations, (str, bytes, bytearray)
        ) and len(operations) > 0:
            return True
    return False


def _has_host_mutation_proof(payload: Mapping[str, Any], name: str) -> bool:
    """Accept only the host-loop marker created after first-party mutation returned."""

    proof = payload.get(_HOST_MUTATION_PROOF_KEY)
    if not isinstance(proof, Mapping):
        return False
    return (
        name == "apply_source_patch"
        and str(proof.get("tool", "")).strip() == name
        and str(proof.get("status", "")).strip() == "APPLIED_BY_HOST_RUNTIME"
    )


def _source_mutation_applied(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only for a successful mutation observation already seen by the host loop."""

    for message in reversed(messages):
        if str(message.get("role", "")).strip().casefold() != "tool":
            continue
        name = str(message.get("name", "")).strip()
        if name not in _MUTATION_TOOLS:
            continue
        payload = _tool_payload(message)
        if payload is None or payload.get("ok") is not True:
            continue
        if _has_host_mutation_proof(payload, name):
            return True
        if _has_applied_patch_receipt(payload):
            return True
        # Older reviewed mutation tools may expose a different receipt envelope.
        # Accept them only when the transport succeeded and no nested result reports
        # an explicit semantic failure. `apply_source_patch` is stricter because its
        # source-patch receipt or host mutation proof is the production custom-coder
        # write contract.
        if name == "apply_source_patch":
            continue
        failed = any(
            str(item.get("status", "")).strip().upper() in _FAILURE_STATUSES
            for item in _walk_mappings(payload)
        )
        if not failed:
            return True
    return False


def _failed_mutation_attempts(
    messages: Sequence[Mapping[str, Any]],
    name: str,
) -> int:
    """Count only explicit host-call failures for one mutation route."""

    failures = 0
    for message in messages:
        if str(message.get("role", "")).strip().casefold() != "tool":
            continue
        if str(message.get("name", "")).strip() != name:
            continue
        payload = _tool_payload(message)
        if payload is not None and payload.get("ok") is False:
            failures += 1
    return failures


def _preferred_visible_mutation(
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Choose a reviewed mutation route without depending on schema ordering."""

    visible = {_tool_name(schema) for schema in tools}
    for name in _MUTATION_PRIORITY:
        if name not in visible:
            continue
        if _failed_mutation_attempts(messages, name) < _MUTATION_FAILURE_LIMIT:
            return name
    return ""


def _force_tool_choice(request: Any, tool_name: str) -> Any:
    from .model_adapters import GenerationRequest

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
    return GenerationRequest(
        messages=request.messages,
        media_paths=request.media_paths,
        response_format=request.response_format,
        response_schema=request.response_schema,
        tools=selected + remainder,
        tool_choice={"type": "function", "function": {"name": tool_name}},
        parallel_tool_calls=False,
        task=getattr(request, "task", ""),
        prompt=getattr(request, "prompt", ""),
        metadata=getattr(request, "metadata", {}),
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

    Prerequisite retrieval remains ``auto`` on the first attempt. If a small model
    answers in prose instead of taking one of the already-authorized causal actions,
    retry exactly once with the first visible frontier action forced. Mutation actions
    are forced by explicit host priority as soon as the causal frontier makes them
    legal, with bounded failover after explicit host-call failures. Final synthesis is
    rejected until a successful mutation observation is present in the transcript.
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

        if request.tool_choice != "auto":
            return self.inner.generate_turn(request)

        visible_mutations = tuple(
            name
            for schema in request.tools
            if (name := _tool_name(schema)) in _MUTATION_TOOLS
        )
        mutation = _preferred_visible_mutation(request.tools, request.messages)
        if mutation:
            forced = _force_tool_choice(request, mutation)
            turn = self.inner.generate_turn(forced)
            if not turn.tool_calls or mutation not in {call.name for call in turn.tool_calls}:
                raise ModelConfigurationError(
                    f"Writable coder did not execute required source-mutation action {mutation!r}; "
                    "refusing a prose-only implementation turn after mutation became causal/legal."
                )
            return turn
        if visible_mutations:
            exhausted = ", ".join(dict.fromkeys(visible_mutations))
            raise ModelConfigurationError(
                "Writable coder exhausted the bounded mutation retry budget for the "
                f"current causal frontier ({exhausted}); refusing to repeat the same "
                "failed source-edit loop."
            )

        # Observation/retrieval frontiers remain model-owned on the first attempt.
        # If the model refuses every visible action and emits prose, that prose is not
        # a valid implementation result: make bounded causal progress by forcing only
        # an action that the outer CausalFrontierAdapter already exposed and fenced.
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
        task=getattr(request, "task", ""),
        prompt=getattr(request, "prompt", ""),
        metadata=getattr(request, "metadata", {}),
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
        # Host custom generation embeds evidence/tool metadata in the user JSON. The
        # explicit phase is stronger terminal intent than incidental strings such as
        # "external MCP" appearing inside that metadata. Use the existing `repair`
        # terminal fact because it is produced only by reviewed source-edit tools;
        # the broader `act/project_changed` goal could terminate on unrelated project
        # mutation and recreate the empty-source-diff failure.
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
