from __future__ import annotations

"""Fast structural runtime checks that run before any production model decode.

These checks deliberately use synthetic tool schemas and fake adapters. They catch
Python/runtime composition regressions (bad wrapper replacement, request-field loss,
unsafe shared state, schema-boundary regressions) before a Colab user spends time
loading multi-gigabyte models. Tool routing itself is owned by the small-model selector
and the normal model tool/observation loop; preflight must not require a second causal
or forced-routing stack.
"""

import json
import sys
import threading
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


def _assert_authoritative_requirement_path() -> None:
    """Require the sole host-owned bounded semantic request path before decode."""

    from . import evidence_first_planning, evidence_request_guard, planning_authority
    from .game_design import GameDesignPlanner
    from .semantic_batching_contract import build_bounded_requirement_catalog

    failures: list[str] = []
    if getattr(GameDesignPlanner.plan, "__mmm_request_contract_guard__", False) is not True:
        failures.append("GameDesignPlanner request freeze/guard")
    if getattr(
        build_bounded_requirement_catalog,
        "__mmm_bounded_semantic_batching__",
        False,
    ) is not True:
        failures.append("bounded semantic catalog builder")
    if getattr(
        evidence_request_guard.build_authoritative_request_catalog,
        "__mmm_bounded_semantic_batching__",
        False,
    ) is not True:
        failures.append("bounded request catalog owner")
    if getattr(
        planning_authority._compile_semantic_catalog,
        "__mmm_bounded_semantic_batching__",
        False,
    ) is not True:
        failures.append("bounded planning semantic compiler")
    if getattr(
        evidence_first_planning._validate_request_catalog,
        "__mmm_approved_requirement_authority__",
        False,
    ) is not True:
        failures.append("approved requirement catalog validator")
    if failures:
        raise RuntimePreflightError(
            "authoritative semantic requirement path is incomplete: " + ", ".join(failures)
        )


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
    from . import small_model_max_agent_contract as small_model

    if getattr(small_model._request_query, "_mmm_structured_terminal_intent", False) is not True:
        raise RuntimePreflightError("small-model selector is not bound to structured terminal intent")
    selector = small_model._request_query(_large_implementation_messages())
    if "implement_module" not in selector:
        raise RuntimePreflightError("structured implementation phase was lost from routing query")


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


def _assert_unordered_retrieval_canonicalization() -> None:
    from .progress_aware_tool_loop import _stable_value, evidence_fingerprint

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
            ("authoritative-requirements", _assert_authoritative_requirement_path),
            ("tool-schema-contracts", _assert_tool_schema_contracts),
            ("routing-intent", _assert_routing_intent_alignment),
            ("generation-concurrency", _assert_generation_concurrency_guards),
            ("retrieval-model-residency", _assert_retrieval_model_residency),
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
