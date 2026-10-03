from __future__ import annotations

"""Finite host-owned loop for reviewed model tools.

The model may gather reviewed observations, but project/source mutation remains
host-owned. Planning, production, repair and completion policy do not live here.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .agent_capability_context import reviewed_mcp_servers_for_model_role
from .model_adapters import GenerationRequest, ModelConfigurationError
from .model_context_budget import bounded_tool_message, fit_messages_to_context
from .tool_transition_registry import reviewed_transition


_HOST_OWNED_MUTATION_EFFECTS = frozenset({
    "project_changed",
    "source_generated",
    "assets_generated",
    "generated",
    "repaired",
    "packaged",
    "work_changed",
})


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name") or "").strip()


def _is_host_owned_mutation(name: str) -> bool:
    transition = reviewed_transition(str(name or "").strip())
    return bool(
        transition is not None
        and transition.effects.intersection(_HOST_OWNED_MUTATION_EFFECTS)
    )


def _call_signature(call: Any) -> tuple[str, str] | None:
    try:
        return (
            str(call.name),
            json.dumps(
                dict(call.arguments),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _assistant_message(turn: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": turn.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments or json.dumps(
                        dict(call.arguments),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for call in turn.tool_calls
        ],
    }


def _evidence_tool_names(
    tools: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    names = {_tool_name(schema) for schema in tools}
    local = tuple(
        name
        for name in ("search_code_rag", "search_project_rag")
        if name in names
    )
    if local:
        return local
    return tuple(name for name in ("external_mcp_call",) if name in names)


def generate_with_reviewed_tools(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    from .grounding_policy import host_baseline_evidence_ready
    from .model_router import (
        _RAG_EVIDENCE_TOOLS,
        _agent_tool_round_limit,
        _execute_tool_waves,
        _parallel_read_call,
        _usable_external_rag_result,
        _usable_rag_result,
    )

    messages = [dict(message) for message in request.messages]
    visible_tools = tuple(request.tools)
    visible_names = frozenset(
        name for schema in visible_tools if (name := _tool_name(schema))
    )
    reviewed_servers = reviewed_mcp_servers_for_model_role(stage, role)
    seen_reads: set[tuple[str, str]] = set()
    evidence_required = bool(
        getattr(router, "_agent_require_fresh_evidence", False)
        and role in {"coder", "coder_safe"}
    )
    evidence_ready = (
        host_baseline_evidence_ready(messages)
        if evidence_required
        else True
    )
    force_evidence = False

    for round_index in range(_agent_tool_round_limit()):
        active_tools = visible_tools
        tool_choice = request.tool_choice
        parallel = request.parallel_tool_calls

        if force_evidence:
            evidence_names = _evidence_tool_names(visible_tools)
            if not evidence_names:
                raise ModelConfigurationError(
                    "FRESH_EVIDENCE_UNAVAILABLE: no reviewed evidence tool remains."
                )
            active_tools = tuple(
                schema
                for schema in visible_tools
                if _tool_name(schema) in evidence_names
            )
            tool_choice = (
                {
                    "type": "function",
                    "function": {"name": evidence_names[0]},
                }
                if len(evidence_names) == 1
                else "required"
            )
            parallel = False

        turn_request = replace(
            request,
            messages=fit_messages_to_context(
                messages,
                config=config,
                tools=active_tools,
            ),
            media_paths=request.media_paths if round_index == 0 else (),
            tools=active_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
        )
        with router._generation_scope(config):
            turn = adapter.generate_turn(turn_request)

        if force_evidence:
            allowed = frozenset(_evidence_tool_names(active_tools))
            if not turn.tool_calls or any(
                call.name not in allowed for call in turn.tool_calls
            ):
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    "Production coder did not honor host-forced RAG tool choice; "
                    f"received {called}."
                )

        if not turn.tool_calls:
            if evidence_required and not evidence_ready:
                force_evidence = True
                continue
            return str(turn.content or "")

        messages.append(_assistant_message(turn))

        def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
            name = str(getattr(call, "name", "") or "").strip()
            if name not in visible_names:
                return call, {
                    "ok": False,
                    "tool": name,
                    "failure_code": "TOOL_NOT_EXPOSED",
                    "error": "Tool is outside the host-reviewed surface.",
                }
            if _is_host_owned_mutation(name):
                return call, {
                    "ok": False,
                    "tool": name,
                    "failure_code": "HOST_OWNED_MUTATION_REQUIRED",
                    "error": (
                        "Project/source mutation is host-owned and cannot be "
                        "executed from the model tool loop."
                    ),
                }

            signature = _call_signature(call)
            if (
                signature is not None
                and _parallel_read_call(call)
                and signature in seen_reads
            ):
                return call, {
                    "ok": False,
                    "tool": name,
                    "failure_code": "DUPLICATE_QUERY",
                    "error": "Equivalent reviewed read call was already attempted.",
                }
            if signature is not None and _parallel_read_call(call):
                seen_reads.add(signature)

            try:
                scoped = getattr(runtime, "call_scoped", None)
                if callable(scoped):
                    result = scoped(
                        stage,
                        name,
                        call.arguments,
                        external_server_ids=reviewed_servers,
                    )
                else:
                    result = runtime.call(stage, name, call.arguments)
                return call, {"ok": True, "tool": name, "result": result}
            except Exception as exc:
                return call, {
                    "ok": False,
                    "tool": name,
                    "failure_code": "TOOL_RUNTIME_ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        executed = _execute_tool_waves(tuple(turn.tool_calls), execute)
        for call, payload in executed:
            result = payload.get("result") if isinstance(payload, Mapping) else None
            if bool(payload.get("ok")):
                if (
                    call.name in _RAG_EVIDENCE_TOOLS
                    and _usable_rag_result(result)
                ):
                    evidence_ready = True
                elif (
                    call.name == "external_mcp_call"
                    and _usable_external_rag_result(call.arguments, result)
                ):
                    evidence_ready = True

            messages.append(dict(bounded_tool_message(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                },
                config=config,
                tools=visible_tools,
            )))

        force_evidence = bool(evidence_required and not evidence_ready)

    raise ModelConfigurationError(
        "AGENT_TOOL_ROUND_LIMIT: reviewed tool conversation exceeded "
        "the host hard cap."
    )


__all__ = ["generate_with_reviewed_tools"]
