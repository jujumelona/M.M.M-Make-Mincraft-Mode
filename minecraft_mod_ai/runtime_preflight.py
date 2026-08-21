from __future__ import annotations

"""Fast structural runtime checks that run before any production model decode.

These checks deliberately use synthetic tool schemas and fake adapters. They are
intended to catch Python/runtime composition regressions (bad wrapper replacement,
unhashable causal state, request-field loss, broken causal progression, unsafe shared
state) before a Colab user spends time loading multi-gigabyte models.
"""

import io
import json
import sys
import threading
from contextlib import redirect_stdout
from typing import Any

_PREFLIGHT_LOCK = threading.RLock()
_PREFLIGHT_DONE = False


class RuntimePreflightError(RuntimeError):
    pass


def _schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class _CaptureAdapter:
    def __init__(self) -> None:
        self.request: Any | None = None

    def generate_turn(self, request: Any) -> Any:
        from .model_adapters import GenerationResponse

        self.request = request
        return GenerationResponse(content="preflight-ok")


def _large_implementation_messages() -> tuple[dict[str, str], ...]:
    payload = {
        "phase": "implement_module",
        "task": "Implement the approved Minecraft/Fabric feature in the current project.",
        # Deliberately larger than the historical 12 KiB tail routing window.
        "research_context": "e" * 20_000,
    }
    return ({"role": "user", "content": json.dumps(payload)},)


def _assert_wrapper_chain() -> None:
    from .model_router import ModelRouter

    current: Any = ModelRouter._generate_with_tools
    required = (
        "_mmm_coder_tool_route_integrity_v1",
        "_mmm_progress_aware_causal_composed",
        "_mmm_writable_coder_fail_closed",
        "_mmm_dynamic_causal_frontier",
    )
    missing = [name for name in required if getattr(current, name, False) is not True]
    if missing:
        raise RuntimePreflightError(
            "final ModelRouter tool-loop composition is missing contracts: "
            + ", ".join(missing)
        )

    seen: set[int] = set()
    depth = 0
    while callable(current):
        marker = id(current)
        if marker in seen:
            raise RuntimePreflightError("ModelRouter tool-loop __wrapped__ chain contains a cycle")
        seen.add(marker)
        depth += 1
        if depth > 64:
            raise RuntimePreflightError("ModelRouter tool-loop wrapper chain is unexpectedly deep")
        current = getattr(current, "__wrapped__", None)


def _assert_tool_schema_contracts() -> None:
    """Require every fail-closed schema/request boundary installed by finalization."""

    from . import external_agent_bridge, external_mcp_router
    from .agent_tool_runtime import AgentToolRuntime
    from .model_adapters import llama_cpp_adapter

    checks = (
        (
            "first-party raw tools/list integrity",
            AgentToolRuntime._list_tools_async,
            "_mmm_raw_mcp_schema_integrity_v1",
        ),
        (
            "first-party schema child-environment binding",
            AgentToolRuntime.tool_schemas,
            "_mmm_mcp_schema_environment_v1",
        ),
        (
            "external provider schema integrity",
            external_agent_bridge._provider_schema,
            "_mmm_external_provider_schema_integrity_v1",
        ),
        (
            "external provider pre-call integrity",
            external_mcp_router.ExternalMCPRouter._initialized_call,
            "_mmm_external_provider_call_integrity_v1",
        ),
        (
            "external schema-to-provider execution binding",
            external_agent_bridge.ExternalAgentBridge.call,
            "_mmm_external_mcp_schema_binding_v1",
        ),
        (
            "external same-scope schema binding serialization",
            external_agent_bridge.ExternalAgentBridge.call,
            "_mmm_external_mcp_scope_serialization_v1",
        ),
        (
            "external model-facing bound-call schema",
            external_agent_bridge.ExternalAgentBridge.tool_schemas,
            "_mmm_external_mcp_bound_schema_v1",
        ),
        (
            "Qwen visible/authorized validation surface",
            llama_cpp_adapter._qwen_tool_generation_response,
            "_mmm_authorized_tool_validation_surface",
        ),
        (
            "reasoning continuation request preservation",
            llama_cpp_adapter._reasoning_continuation_request,
            "_mmm_tool_validation_continuation",
        ),
    )
    missing = [
        label
        for label, target, marker in checks
        if getattr(target, marker, False) is not True
    ]
    router_class = external_mcp_router.ExternalMCPRouter
    if (
        getattr(router_class, "_mmm_external_mcp_bound_invoke_v1", False) is not True
        or not callable(getattr(router_class, "invoke_bound", None))
    ):
        missing.append("exact external MCP bound-provider invocation")
    if missing:
        raise RuntimePreflightError(
            "final tool/schema runtime is missing fail-closed contracts: "
            + ", ".join(missing)
        )


def _assert_routing_intent_alignment() -> None:
    from .causal_frontier_adapter import _query as causal_query
    from .causal_tool_frontier_contract import goals_for_query
    from . import small_model_max_agent_contract as small_model

    if getattr(small_model._request_query, "_mmm_structured_terminal_intent", False) is not True:
        raise RuntimePreflightError("small-model selector is not bound to structured terminal intent")
    messages = _large_implementation_messages()
    causal = causal_query(messages)
    selector = small_model._request_query(messages)
    if selector != causal:
        raise RuntimePreflightError("small-model and causal routing queries diverged")
    if "implement_module" not in selector:
        raise RuntimePreflightError("structured implementation phase was lost from routing query")
    goals = tuple(goals_for_query(selector))
    if goals != ("repair",):
        raise RuntimePreflightError(
            f"structured implementation request routed to {goals!r}, not repair"
        )


def _assert_generation_concurrency_guards() -> None:
    from .custom_module_generator import CustomModuleGenerator
    from .project_index import ProjectIndex

    if getattr(CustomModuleGenerator.generate, "_mmm_instance_generation_serialized", False) is not True:
        raise RuntimePreflightError("shared CustomModuleGenerator is missing its per-instance lock")
    for method_name in (
        "update_files",
        "write_manifest",
        "manifest",
        "manifest_receipt",
        "select",
        "select_page",
    ):
        method = getattr(ProjectIndex, method_name)
        if getattr(method, "_mmm_snapshot_locked", False) is not True:
            raise RuntimePreflightError(
                f"ProjectIndex.{method_name} is outside the shared snapshot lock"
            )


def _assert_retrieval_model_residency() -> None:
    from .model_router import ModelRouter

    if getattr(ModelRouter.embed, "_mmm_resident_embedding_adapter", False) is not True:
        raise RuntimePreflightError(
            "ModelRouter.embed would reconstruct the embedding adapter per RAG batch"
        )
    if getattr(ModelRouter.rerank, "_mmm_resident_reranker_adapter", False) is not True:
        raise RuntimePreflightError(
            "ModelRouter.rerank would reconstruct the reranker adapter per query"
        )


def _assert_repair_causal_progression() -> None:
    from .causal_tool_graph import executable_frontier, verified_state_from_messages

    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    initial = executable_frontier(
        schemas,
        state=frozenset({"workspace_bound"}),
        goals=("repair",),
        limit=3,
        max_depth=8,
    )
    if initial != ("search_code_rag",):
        raise RuntimePreflightError(
            f"repair causal entry frontier drifted: expected search_code_rag, got {initial!r}"
        )

    payload = {
        "ok": True,
        "tool": "search_code_rag",
        "result": {
            "receipt": {
                "result_count": 1,
                "coverage_score": 1.0,
                "relevance_score": 1.0,
            }
        },
    }
    state = verified_state_from_messages(
        (
            {
                "role": "tool",
                "name": "search_code_rag",
                "content": json.dumps(payload, separators=(",", ":")),
            },
        ),
        schemas,
    )
    after_evidence = executable_frontier(
        schemas,
        state=state,
        goals=("repair",),
        limit=3,
        max_depth=8,
    )
    if after_evidence != ("apply_source_patch",):
        raise RuntimePreflightError(
            "repair causal mutation frontier is unreachable after valid RAG evidence: "
            f"{after_evidence!r}"
        )


def _assert_per_turn_adapter() -> None:
    from .causal_frontier_adapter import CausalFrontierAdapter, clear_current_frontier
    from .model_adapters import GenerationRequest

    messages = _large_implementation_messages()
    capture = _CaptureAdapter()
    request = GenerationRequest(
        messages=messages,
        tools=(_schema("search_code_rag"), _schema("apply_source_patch")),
        tool_choice="auto",
        task="sentinel-task",
        prompt="sentinel-prompt",
        metadata={"sentinel": "metadata"},
    )
    adapter = CausalFrontierAdapter(
        capture,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
    )
    clear_current_frontier()
    try:
        with redirect_stdout(io.StringIO()):
            response = adapter.generate_turn(request)
    finally:
        clear_current_frontier()
    if response.content != "preflight-ok" or capture.request is None:
        raise RuntimePreflightError("per-turn causal adapter did not complete its synthetic turn")
    forwarded = capture.request
    if forwarded.task != request.task or forwarded.prompt != request.prompt:
        raise RuntimePreflightError("per-turn causal adapter dropped task/prompt fields")
    if dict(forwarded.metadata) != dict(request.metadata):
        raise RuntimePreflightError("per-turn causal adapter dropped request metadata")
    names = tuple(
        str(schema.get("function", {}).get("name", "")) for schema in forwarded.tools
    )
    if names != ("search_code_rag",):
        raise RuntimePreflightError(f"unexpected initial per-turn causal surface: {names!r}")


def _assert_compaction_clone() -> None:
    from . import small_model_compacting_adapter as compaction_module
    from .model_adapters import GenerationRequest

    capture = _CaptureAdapter()
    adapter = compaction_module.CompactingAdapter(capture)
    request = GenerationRequest(
        messages=({"role": "user", "content": "preflight"},),
        task="sentinel-task",
        prompt="sentinel-prompt",
        metadata={"sentinel": "metadata"},
    )
    original = compaction_module.compact_messages
    compaction_module.compact_messages = lambda messages: (
        *tuple(messages),
        {"role": "system", "content": "synthetic compacted context"},
    )
    try:
        adapter.generate_turn(request)
    finally:
        compaction_module.compact_messages = original
    if capture.request is None:
        raise RuntimePreflightError("compaction adapter did not forward its synthetic request")
    forwarded = capture.request
    if forwarded.task != request.task or forwarded.prompt != request.prompt:
        raise RuntimePreflightError("compaction adapter dropped task/prompt fields")
    if dict(forwarded.metadata) != dict(request.metadata):
        raise RuntimePreflightError("compaction adapter dropped request metadata")


def _assert_unordered_retrieval_canonicalization() -> None:
    from .retrieval_progress import _stable_value, evidence_fingerprint

    stable = _stable_value({"facts": {"b", "a"}}, drop_volatile=False)
    if stable != {"facts": ["a", "b"]}:
        raise RuntimePreflightError(f"set-valued retrieval state is not canonical: {stable!r}")
    left = evidence_fingerprint({"facts": {"a", "b"}})
    right = evidence_fingerprint({"facts": frozenset(("b", "a"))})
    if not left or left != right:
        raise RuntimePreflightError("equivalent unordered retrieval evidence fingerprints diverged")


def run_runtime_preflight() -> None:
    """Fail fast on structural agent regressions before expensive model loading."""

    global _PREFLIGHT_DONE
    if _PREFLIGHT_DONE:
        return
    with _PREFLIGHT_LOCK:
        if _PREFLIGHT_DONE:
            return
        checks = (
            ("wrapper-chain", _assert_wrapper_chain),
            ("tool-schema-contracts", _assert_tool_schema_contracts),
            ("routing-intent", _assert_routing_intent_alignment),
            ("generation-concurrency", _assert_generation_concurrency_guards),
            ("retrieval-model-residency", _assert_retrieval_model_residency),
            ("repair-causal-progression", _assert_repair_causal_progression),
            ("per-turn-adapter", _assert_per_turn_adapter),
            ("compaction-clone", _assert_compaction_clone),
            ("retrieval-canonicalization", _assert_unordered_retrieval_canonicalization),
        )
        for name, check in checks:
            try:
                check()
            except RuntimePreflightError:
                raise
            except BaseException as exc:
                raise RuntimePreflightError(
                    f"runtime preflight {name!r} crashed: {type(exc).__name__}: {exc}"
                ) from exc
        _PREFLIGHT_DONE = True
        # MCP stdio reserves stdout for JSON-RPC frames. Diagnostics must never
        # write there, including package-import preflight success messages.
        print("runtime preflight: PASS", file=sys.stderr, flush=True)


__all__ = ["RuntimePreflightError", "run_runtime_preflight"]