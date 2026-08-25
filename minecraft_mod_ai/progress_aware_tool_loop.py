from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Mapping

from .agent_intent import implementation_requested
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    completion_boundary_kind,
    mark_context_recovery_exhausted,
)
from .model_adapters import GenerationRequest, ModelConfigurationError
from .model_context_budget import (
    bounded_tool_message,
    emergency_fit_messages,
    fit_messages_to_context,
    request_message_budget,
)
from .retrieval_progress import RetrievalDecision, RetrievalObservation, RetrievalProgress
from .source_mutation_contract import mutation_history_applied


def _replace_live_messages(
    messages: list[dict[str, Any]],
    fitted: tuple[Mapping[str, Any], ...],
) -> bool:
    replacement = [dict(message) for message in fitted]
    if replacement == messages:
        return False
    messages[:] = replacement
    return True


def _generate_turn_with_context_recovery(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    media_paths: tuple[Any, ...],
    tool_choice: Any,
    parallel_tool_calls: bool,
) -> Any:
    """Fit one live agent turn and retry context pressure exactly once."""

    fitted = fit_messages_to_context(messages, config=config, tools=request.tools)
    _replace_live_messages(messages, fitted)
    turn_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    try:
        with router._generation_scope(config):
            return adapter.generate_turn(turn_request)
    except BaseException as exc:
        if completion_boundary_kind(exc) != CONTEXT_PRESSURE:
            raise

        emergency_budget = min(
            40 * 1024,
            request_message_budget(config, request.tools),
        )
        emergency = emergency_fit_messages(
            messages,
            budget_bytes=emergency_budget,
        )
        if not _replace_live_messages(messages, emergency):
            mark_context_recovery_exhausted(exc)
            raise

        retry_request = replace(
            turn_request,
            messages=tuple(messages),
            media_paths=media_paths,
        )
        print(
            "agent context: deterministic overflow recovery",
            f"messages={len(messages)}",
            f"budget_bytes={emergency_budget}",
            flush=True,
        )
        try:
            with router._generation_scope(config):
                return adapter.generate_turn(retry_request)
        except BaseException as retry_exc:
            if completion_boundary_kind(retry_exc) == CONTEXT_PRESSURE:
                mark_context_recovery_exhausted(exc)
                raise exc from retry_exc
            raise


def generate_with_tools(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    """Run the canonical host-planned retrieve/inspect/mutate/verify loop.

    One causal ledger computes the legal executable frontier every round. The host then
    chooses one action class before model generation. Mutation frontiers collapse to one
    deterministic writable tool and the model generates only that tool's arguments.
    Every tool call is checked again against the published frontier immediately before
    execution, so stale authorized tools are observations of invalid model behavior,
    never executable actions.
    """
    from .agent_capability_context import reviewed_mcp_servers_for_model_role, skills_for_tool
    from .causal_frontier_adapter import CausalFrontierAdapter, FrontierExecutionGate
    from .grounding_policy import host_baseline_evidence_ready
    from .model_router import (
        _RAG_EVIDENCE_TOOLS,
        _agent_tool_round_limit,
        _execute_tool_waves,
        _external_rag_capability,
        _tool_schema_names,
        _usable_external_rag_result,
        _usable_rag_result,
    )

    messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
    exposed_tools = frozenset(_tool_schema_names(request.tools))
    progress = RetrievalProgress()
    forced_rag_tool: str | None = None
    round_index = 0
    round_limit = _agent_tool_round_limit()
    host_grounded = host_baseline_evidence_ready(request.messages)
    require_rag = bool(
        router._agent_require_fresh_evidence
        and not host_grounded
        and role in {"coder", "coder_safe"}
        and exposed_tools & _RAG_EVIDENCE_TOOLS
    )
    implementation_requires_mutation = bool(
        role in {"coder", "coder_safe"}
        and stage == "generation"
        and implementation_requested(request.messages)
    )
    reviewed_external_servers = reviewed_mcp_servers_for_model_role(stage, role)

    execution_gate = FrontierExecutionGate()
    causal_adapter = CausalFrontierAdapter(
        adapter,
        stage=stage,
        role=role,
        require_fresh_evidence=bool(router._agent_require_fresh_evidence),
        execution_gate=execution_gate,
        authorized_surface=request.tools,
        request_template=request,
    )

    while True:
        if round_limit is not None and round_index >= round_limit:
            if require_rag and not progress.has_fresh_evidence:
                raise ModelConfigurationError(
                    "Agent reached the explicit tool-round limit before required "
                    "evidence became available."
                )
            if implementation_requires_mutation and not mutation_history_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder reached the explicit tool-round limit before a "
                    "reviewed source mutation was applied; refusing a prose-only implementation."
                )
            return _finalize_without_tools(
                router,
                config,
                causal_adapter,
                request,
                messages,
                instruction=(
                    f"The explicitly configured host tool limit was reached after {round_limit} rounds. "
                    "Do not call more tools. Return the final answer using only observations already present."
                ),
                empty_error="Agent returned an empty final response at the explicit tool-round limit.",
            )

        tool_choice = request.tool_choice
        parallel_tool_calls = request.parallel_tool_calls
        if forced_rag_tool is not None:
            tool_choice = {"type": "function", "function": {"name": forced_rag_tool}}
            parallel_tool_calls = False

        turn = _generate_turn_with_context_recovery(
            router,
            config=config,
            adapter=causal_adapter,
            request=request,
            messages=messages,
            media_paths=request.media_paths if round_index == 0 else (),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        if not turn.tool_calls:
            content = turn.content.strip()
            if not content:
                raise ModelConfigurationError("Tool-capable model returned an empty final response.")
            if forced_rag_tool is not None:
                raise ModelConfigurationError(
                    f"Host-selected RAG action {forced_rag_tool!r} returned prose instead of one "
                    "schema-valid action; tool-name selection is never retried."
                )
            if require_rag and not progress.has_fresh_evidence:
                forced_rag_tool = progress.next_untried_internal_tool(
                    exposed_tools,
                    preferred=("search_code_rag", "search_project_rag"),
                )
                if forced_rag_tool is None:
                    raise ModelConfigurationError(
                        "Required production evidence is unavailable: every host-forceable "
                        "RAG source was already attempted without novel usable evidence, and "
                        "the model selected no other reviewed retrieval route."
                    )
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "system",
                            "content": (
                                "The host selected the next evidence action. Generate only its "
                                "schema-valid arguments; do not choose another tool name."
                            ),
                        },
                    ]
                )
                round_index += 1
                continue
            if implementation_requires_mutation and not mutation_history_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder returned a final prose answer before a reviewed source mutation "
                    "was applied; implementation completion requires a real source diff."
                )
            return content

        if forced_rag_tool is not None:
            if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_rag_tool:
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    f"Host-selected RAG action {forced_rag_tool!r} produced {called}; "
                    "refusing tool-name reselection."
                )
            forced_rag_tool = None

        messages.append(
            {
                "role": "assistant",
                "content": turn.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments
                            or json.dumps(
                                dict(call.arguments),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for call in turn.tool_calls
                ],
            }
        )

        def is_retrieval(call: Any) -> bool:
            return call.name in _RAG_EVIDENCE_TOOLS or (
                call.name == "external_mcp_call"
                and bool(_external_rag_capability(call.arguments))
            )

        def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
            route_metadata: dict[str, Any] = {
                "skills": list(skills_for_tool(stage, call.name, model_role=role))
            }
            if call.name == "external_mcp_call":
                capability = str(call.arguments.get("capability", "")).strip()
                if capability:
                    route_metadata["external_mcp_capability"] = capability
            try:
                visible = execution_gate.visible_names()
                if visible is not None and call.name not in visible:
                    raise ModelConfigurationError(
                        "IllegalAction: tool is authorized for the role but illegal at the "
                        f"current causal frontier: {call.name!r}; visible={','.join(visible)}"
                    )
                if call.name not in exposed_tools:
                    raise ModelConfigurationError(
                        f"Agent attempted hidden tool {call.name!r} outside its reviewed role routes for {role!r}/{stage!r}."
                    )
                if is_retrieval(call):
                    decision = progress.begin(call.name, call.arguments)
                    if decision is not RetrievalDecision.EXECUTE:
                        return call, {
                            "ok": False,
                            "tool": call.name,
                            **route_metadata,
                            "error": "RetrievalNoProgress: "
                            + (
                                "equivalent query already attempted for this evidence need"
                                if decision is RetrievalDecision.DUPLICATE_QUERY
                                else "retrieval source reached a repeated-evidence fixed point"
                            ),
                        }
                scoped_call = getattr(runtime, "call_scoped", None)
                if callable(scoped_call):
                    result = scoped_call(
                        stage,
                        call.name,
                        call.arguments,
                        external_server_ids=reviewed_external_servers,
                    )
                elif call.name.startswith("external_mcp_"):
                    raise ModelConfigurationError(
                        "External MCP execution requires a role-scoped agent runtime."
                    )
                else:
                    result = runtime.call(stage, call.name, call.arguments)
                payload: Mapping[str, Any] = {
                    "ok": True,
                    "tool": call.name,
                    **route_metadata,
                    "result": result,
                }
                if call.name == "apply_source_patch":
                    payload = {
                        **payload,
                        "_mmm_source_mutation": {
                            "tool": "apply_source_patch",
                            "status": "APPLIED_BY_HOST_RUNTIME",
                        },
                    }
            except Exception as exc:
                payload = {
                    "ok": False,
                    "tool": call.name,
                    **route_metadata,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return call, payload

        executed = _execute_tool_waves(tuple(turn.tool_calls), execute)
        retrieval_no_progress = False
        weak_retrieval = False
        for call, payload in executed:
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
            messages.append(
                dict(
                    bounded_tool_message(
                        tool_message,
                        config=config,
                        tools=request.tools,
                    )
                )
            )
            if not is_retrieval(call):
                continue
            if not bool(payload.get("ok")):
                if str(payload.get("error", "")).startswith("RetrievalNoProgress:"):
                    retrieval_no_progress = True
                continue
            if call.name in _RAG_EVIDENCE_TOOLS:
                usable = _usable_rag_result(payload.get("result"))
            else:
                usable = _usable_external_rag_result(
                    call.arguments,
                    payload.get("result"),
                )
            observation = progress.observe(payload.get("result"), usable=usable)
            if observation is RetrievalObservation.DUPLICATE_EVIDENCE:
                retrieval_no_progress = True
            elif observation is RetrievalObservation.WEAK:
                weak_retrieval = True

        if retrieval_no_progress and progress.has_fresh_evidence:
            if implementation_requires_mutation and not mutation_history_applied(messages):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Retrieval has converged and usable evidence is already available. "
                            "Do not finalize: proceed to the host-selected source-mutation action."
                        ),
                    }
                )
                round_index += 1
                continue
            return _finalize_without_tools(
                router,
                config,
                causal_adapter,
                request,
                messages,
                instruction=(
                    "Retrieval reached a no-progress fixed point after usable evidence was already gathered. "
                    "Do not call more tools. Return the final answer from existing observations."
                ),
                empty_error="Agent returned an empty final response after retrieval convergence.",
            )

        if require_rag and not progress.has_fresh_evidence and (
            weak_retrieval or retrieval_no_progress
        ):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The latest retrieval added no usable novel evidence. Inspect its receipt and "
                        "choose a materially different reviewed retrieval route only if one can add new "
                        "information. Repeating an equivalent query or repeated-evidence source is not progress."
                    ),
                }
            )
        round_index += 1


def _finalize_without_tools(
    router: Any,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    *,
    instruction: str,
    empty_error: str,
) -> str:
    final_messages = [*messages, {"role": "system", "content": instruction}]
    final_request = replace(
        request,
        messages=tuple(final_messages),
        media_paths=(),
        tools=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )
    final_messages_mutable = [dict(message) for message in final_messages]
    final_turn = _generate_turn_with_context_recovery(
        router,
        config=config,
        adapter=adapter,
        request=final_request,
        messages=final_messages_mutable,
        media_paths=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )
    if final_turn.tool_calls:
        raise ModelConfigurationError("Agent emitted tool calls after the host disabled tools.")
    content = final_turn.content.strip()
    if not content:
        raise ModelConfigurationError(empty_error)
    return content
