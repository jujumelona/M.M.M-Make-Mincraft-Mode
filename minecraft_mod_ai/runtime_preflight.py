from __future__ import annotations

"""Fast structural runtime checks that run before any production model decode.

These checks deliberately use synthetic tool schemas and fake adapters. They catch
Python/runtime composition regressions before a Colab user spends time loading
multi-gigabyte models. Mandatory planning is compiler-owned and is verified directly;
preflight must never depend on a runtime planning monkey-patch.
"""

import inspect
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
    """Require the single compiler-owned semantic/design path before decode."""

    from . import agentic_research_game_design, planning_authority
    from .planning_pipeline import PlanningPipeline

    failures: list[str] = []
    if planning_authority.build_authoritative_request_catalog.__module__ != planning_authority.__name__:
        failures.append("request catalog owner")
    if planning_authority.authoritative_request_scope.__module__ != planning_authority.__name__:
        failures.append("request authority state owner")
    if agentic_research_game_design.generate_sectioned_game_design.__module__ != agentic_research_game_design.__name__:
        failures.append("host game-design compiler")
    if agentic_research_game_design.validate_ready_design.__module__ != agentic_research_game_design.__name__:
        failures.append("host design readiness validator")
    if PlanningPipeline._semantic_design.__module__ != "minecraft_mod_ai.planning_pipeline":
        failures.append("canonical planning pipeline")
    for target, label in (
        (planning_authority.build_authoritative_request_catalog, "request catalog owner"),
        (agentic_research_game_design.generate_sectioned_game_design, "host game-design compiler"),
        (PlanningPipeline._semantic_design, "canonical planning pipeline"),
    ):
        if getattr(target, "__wrapped__", None) is not None:
            failures.append(label + " is runtime wrapped")
    if failures:
        raise RuntimePreflightError(
            "compiler-owned planning authority is incomplete: " + ", ".join(failures)
        )


def _assert_tool_schema_contracts() -> None:
    """Verify the direct fail-closed tool/schema boundary without wrapper markers."""

    from .agent_tool_runtime import AgentToolRuntime
    from .external_agent_bridge import (
        TOOL_NAMES as EXTERNAL_TOOL_NAMES,
        ExternalAgentBridge,
    )
    from .external_mcp_router import ExternalMCPRouter
    from .model_adapters import llama_cpp_adapter

    required_callables = (
        ("first-party tools/list owner", AgentToolRuntime._list_tools_async),
        ("first-party schema owner", AgentToolRuntime.tool_schemas),
        ("external provider schema owner", ExternalAgentBridge.tool_schemas),
        ("external provider call owner", ExternalAgentBridge.call),
        ("external MCP router", ExternalMCPRouter.invoke),
        ("external MCP initialized call", ExternalMCPRouter._initialized_call),
        ("native tool schema map", llama_cpp_adapter._request_tool_schema_map),
        ("native exact visible-tool binding", llama_cpp_adapter._exact_model_visible_tools),
    )
    missing = [label for label, target in required_callables if not callable(target)]

    schemas = ExternalAgentBridge.tool_schemas("generation")
    names: list[str] = []
    for schema in schemas:
        if not isinstance(schema, dict):
            missing.append("external model-facing schema object")
            continue
        function = schema.get("function")
        if not isinstance(function, dict):
            missing.append("external model-facing function schema")
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            missing.append("external model-facing tool name")
            continue
        names.append(name)
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            missing.append(f"{name} object parameter schema")
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            missing.append(f"{name} parameter properties")
            continue
        for host_owned in ("minecraft_version", "loader", "mappings"):
            raw = properties.get(host_owned)
            if not isinstance(raw, dict) or "type" in raw:
                missing.append(f"{name} host-owned {host_owned} binding")

    if set(names) != set(EXTERNAL_TOOL_NAMES) or len(names) != len(set(names)):
        missing.append("exact external model-facing tool set")

    if missing:
        raise RuntimePreflightError(
            "direct tool/schema runtime is incomplete: " + ", ".join(sorted(set(missing)))
        )

def _assert_routing_intent_alignment() -> None:
    from . import small_model_max_agent_contract as small_model

    selector = small_model._request_query(_large_implementation_messages())
    if "implement_module" not in selector:
        raise RuntimePreflightError("structured implementation phase was lost from routing query")


def _assert_generation_concurrency_guards() -> None:
    from .custom_module_generator import CustomModuleGenerator
    from .project_index import ProjectIndex

    generator_source = inspect.getsource(CustomModuleGenerator.generate)
    if "with project_write_lock(root):" not in generator_source:
        raise RuntimePreflightError(
            "CustomModuleGenerator does not directly own its project write lock"
        )

    for method_name in (
        "update_files",
        "write_manifest",
        "manifest",
        "manifest_receipt",
        "select",
        "select_page",
    ):
        method = getattr(ProjectIndex, method_name)
        if getattr(method, "__wrapped__", None) is None:
            raise RuntimePreflightError(
                f"ProjectIndex.{method_name} is outside the source-owned snapshot lock"
            )


def _assert_retrieval_model_residency() -> None:
    from .model_router import ModelRouter

    embed_source = inspect.getsource(ModelRouter.embed)
    rerank_source = inspect.getsource(ModelRouter.rerank)
    if "_embedding_adapters" not in embed_source:
        raise RuntimePreflightError(
            "ModelRouter.embed would reconstruct the embedding adapter per RAG batch"
        )
    if "_reranker_adapters" not in rerank_source:
        raise RuntimePreflightError(
            "ModelRouter.rerank would reconstruct the reranker adapter per query"
        )


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
            ("tool-loop-integrity", _assert_wrapper_chain),
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
        print("runtime preflight: PASS", file=sys.stderr, flush=True)


__all__ = ["RuntimePreflightError", "run_runtime_preflight"]