from __future__ import annotations

import json
from typing import Any, Mapping

from .model_adapters import GenerationRequest, ModelConfigurationError
from .retrieval_progress import RetrievalDecision, RetrievalObservation, RetrievalProgress


def generate_with_tools(
    router: Any,
    *,
    adapter: Any,
    request: GenerationRequest,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    """Run retrieve/act/observe with semantic retrieval progress.

    The round budget is only the final liveness guard. Mandatory evidence is
    satisfied by validated host grounding when present; otherwise the host may
    force each internal RAG source at most once. Supplemental retrieval remains
    model-driven and repeated queries/evidence never count as progress.
    """
    from .agent_capability_context import reviewed_mcp_servers_for_model_role, skills_for_tool
    from .coder_tool_route_integrity_contract import (
        _is_implementation_request,
        _source_mutation_applied,
    )
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
    forced_rag_attempts = 0
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
        and _is_implementation_request(request.messages)
    )
    reviewed_external_servers = reviewed_mcp_servers_for_model_role(stage, role)

    while True:
        if round_index >= round_limit:
            if require_rag and not progress.has_fresh_evidence:
                raise ModelConfigurationError(
                    "Agent reached the hard tool-round liveness guard before required "
                    "evidence became available. Retrieval progress should normally "
                    "terminate earlier."
                )
            if implementation_requires_mutation and not _source_mutation_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder reached the hard tool-round liveness guard before a "
                    "reviewed source mutation was applied; refusing a prose-only implementation."
                )
            return _finalize_without_tools(
                adapter,
                request,
                messages,
                instruction=(
                    f"The host tool liveness guard was reached after {round_limit} rounds. "
                    "Do not call more tools. Return the final answer using only observations already present."
                ),
                empty_error="Agent returned an empty final response at the hard tool-round guard.",
            )

        tool_choice = request.tool_choice
        parallel_tool_calls = request.parallel_tool_calls
        if forced_rag_tool is not None:
            tool_choice = {"type": "function", "function": {"name": forced_rag_tool}}
            parallel_tool_calls = False
        turn_request = GenerationRequest(
            messages=messages,
            media_paths=request.media_paths if round_index == 0 else (),
            response_format=request.response_format,
            response_schema=request.response_schema,
            tools=request.tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            task=getattr(request, "task", ""),
            prompt=getattr(request, "prompt", ""),
            metadata=getattr(request, "metadata", {}),
        )
        turn = adapter.generate_turn(turn_request)

        if not turn.tool_calls:
            content = turn.content.strip()
            if not content:
                raise ModelConfigurationError("Tool-capable model returned an empty final response.")
            if forced_rag_tool is not None:
                forced_rag_attempts += 1
                if forced_rag_attempts >= 2:
                    raise ModelConfigurationError(
                        f"Production coder did not honor host-forced RAG tool choice {forced_rag_tool!r} "
                        "after two bounded attempts."
                    )
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": f"Call the required function {forced_rag_tool} exactly once now. Do not answer in prose.",
                    },
                ])
                round_index += 1
                continue
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
                forced_rag_attempts = 0
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": (
                            f"Baseline production evidence is still required. Call {forced_rag_tool} "
                            "exactly once with a concrete query for the current implementation need. "
                            "After its observation, choose any further retrieval only if it can add "
                            "materially new evidence; do not repeat exhausted queries."
                        ),
                    },
                ])
                round_index += 1
                continue
            if implementation_requires_mutation and not _source_mutation_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder returned a final prose answer before a reviewed source mutation "
                    "was applied; implementation completion requires a real source diff."
                )
            return content

        if forced_rag_tool is not None:
            if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_rag_tool:
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    f"Production coder violated host-forced RAG tool choice {forced_rag_tool!r}; received {called}."
                )
            forced_rag_tool = None
            forced_rag_attempts = 0

        messages.append({
            "role": "assistant",
            "content": turn.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or json.dumps(
                            dict(call.arguments), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
                for call in turn.tool_calls
            ],
        })

        def is_retrieval(call: Any) -> bool:
            return call.name in _RAG_EVIDENCE_TOOLS or (
                call.name == "external_mcp_call" and bool(_external_rag_capability(call.arguments))
            )

        def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
            route_metadata: dict[str, Any] = {
                "skills": list(skills_for_tool(stage, call.name, model_role=role))
            }
            if call.name == "external_mcp_call":
                capability = str(call.arguments.get("capability", "")).strip()
                if capability:
                    route_metadata["external_mcp_capability"] = capability
            if is_retrieval(call):
                decision = progress.begin(call.name, call.arguments)
                if decision is not RetrievalDecision.EXECUTE:
                    return call, {
                        "ok": False,
                        "tool": call.name,
                        **route_metadata,
                        "error": "RetrievalNoProgress: " + (
                            "equivalent query already attempted for this evidence need"
                            if decision is RetrievalDecision.DUPLICATE_QUERY
                            else "retrieval source reached a repeated-evidence fixed point"
                        ),
                    }
            try:
                if call.name not in exposed_tools:
                    raise ModelConfigurationError(
                        f"Agent attempted hidden tool {call.name!r} outside its reviewed role routes for {role!r}/{stage!r}."
                    )
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
                # This proof is host-owned: first-party apply_source_patch raises on
                # invalid/no-op/failed transactions, so reaching here means a real
                # staged source mutation returned successfully. Keep the proof beside
                # the potentially size-bounded tool result so result truncation cannot
                # erase the implementation completion fact.
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
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            })
            if not is_retrieval(call):
                continue
            if not bool(payload.get("ok")):
                if str(payload.get("error", "")).startswith("RetrievalNoProgress:"):
                    retrieval_no_progress = True
                continue
            if call.name in _RAG_EVIDENCE_TOOLS:
                usable = _usable_rag_result(payload.get("result"))
            else:
                usable = _usable_external_rag_result(call.arguments, payload.get("result"))
            observation = progress.observe(
                call.name,
                call.arguments,
                payload.get("result"),
                usable=usable,
            )
            if observation is RetrievalObservation.DUPLICATE_EVIDENCE:
                retrieval_no_progress = True
            elif observation is RetrievalObservation.WEAK:
                weak_retrieval = True

        if retrieval_no_progress and progress.has_fresh_evidence:
            if implementation_requires_mutation and not _source_mutation_applied(messages):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Retrieval has converged and usable evidence is already available. "
                            "Do not finalize: proceed to the reviewed source-mutation action for "
                            "the implementation now."
                        ),
                    }
                )
                round_index += 1
                continue
            return _finalize_without_tools(
                adapter,
                request,
                messages,
                instruction=(
                    "Retrieval reached a no-progress fixed point after usable evidence was already gathered. "
                    "Do not call more tools. Return the final answer from existing observations."
                ),
                empty_error="Agent returned an empty final response after retrieval convergence.",
            )

        if require_rag and not progress.has_fresh_evidence and (weak_retrieval or retrieval_no_progress):
            messages.append({
                "role": "system",
                "content": (
                    "The latest retrieval added no usable novel evidence. Inspect its receipt and "
                    "choose a materially different reviewed retrieval route only if one can add new "
                    "information. Repeating an equivalent query or repeated-evidence source is not progress."
                ),
            })
        round_index += 1


def _finalize_without_tools(
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    *,
    instruction: str,
    empty_error: str,
) -> str:
    final_request = GenerationRequest(
        messages=[*messages, {"role": "system", "content": instruction}],
        media_paths=(),
        response_format=request.response_format,
        response_schema=request.response_schema,
        tools=(),
        tool_choice=None,
        parallel_tool_calls=False,
        task=getattr(request, "task", ""),
        prompt=getattr(request, "prompt", ""),
        metadata=getattr(request, "metadata", {}),
    )
    final_turn = adapter.generate_turn(final_request)
    if final_turn.tool_calls:
        raise ModelConfigurationError("Agent emitted tool calls after the host disabled tools.")
    content = final_turn.content.strip()
    if not content:
        raise ModelConfigurationError(empty_error)
    return content
