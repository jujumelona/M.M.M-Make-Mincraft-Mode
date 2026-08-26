from __future__ import annotations

"""Late integration policy that maximizes one frozen small agent model.

This module does not add another generative model or another orchestration owner.
It composes the existing MMM contracts at their established boundaries:

* role/security filtering remains authoritative, then one query-specific tool selector
  reduces the action surface seen by the local model;
* the existing ProjectRAGIndex remains the only code-RAG implementation, while
  semantic retrieval and reranking are enabled when the index can support them;
* repair calls remain fresh/isolated and receive a lossless exact-fact ledger plus
  host-verified failure memory;
* Best-of-N generation and verifier scoring remain owned by
  agentic_optimization_contract.

All adaptations are inference/runtime only. Model weights are never trained or
modified here.
"""

import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from typing import Any

from .agent_intent import is_implementation_intent, structured_user_intent

_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:$<>/-]{1,127}|[가-힣]{2,}")
_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.@+-]+)(?![A-Za-z0-9_])"
)
_SHA = re.compile(r"(?:sha256:)?\b[0-9a-fA-F]{64}\b")
_RESOURCE_ID = re.compile(r"\b[a-z0-9_.-]+:[a-z0-9_./-]+\b")
_VERSION = re.compile(r"\b(?:\d+\.){1,3}\d+(?:[-+._][A-Za-z0-9]+)*\b")
_RAG_ROUTER: ContextVar[Any | None] = ContextVar("mmm_small_agent_rag_router", default=None)
_FAILURE_LOCK = threading.RLock()
_SOURCE_MUTATION_PRIORITY = (
    "apply_source_edit",
    "apply_source_patch",
    "apply_java_operations",
    "repair_project",
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _tool_document(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return "\n".join(
        (
            str(function.get("name", "")),
            str(function.get("description", "")),
            json.dumps(function.get("parameters", {}), ensure_ascii=False, sort_keys=True),
        )
    )


def _request_query(messages: Sequence[Mapping[str, Any]]) -> str:
    """Project only terminal user intent into the tool-retrieval query."""

    return structured_user_intent(messages)


def select_tool_schemas(
    router: Any,
    *,
    role: str,
    query: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    require_fresh_evidence: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Retrieve one stable action surface from an already security-filtered tool set.

    Security/stage/Skill filtering MUST run before this function. It never expands
    the supplied set. A CPU reranker is used when available; deterministic lexical
    scoring is the fallback. Mandatory capabilities are preserved inside this one
    selector instead of being re-routed by a second per-turn policy.
    """

    tools = tuple(tool_schemas)
    top_k = _env_int("MMM_SMALL_MODEL_TOOL_TOP_K", 5, minimum=3, maximum=8)
    if len(tools) <= top_k:
        return tools

    query_tokens = _tokens(query)
    available_names = {_tool_name(schema) for schema in tools}
    mandatory_names: list[str] = []
    if require_fresh_evidence and role in {"coder", "coder_safe"}:
        for name in ("search_code_rag", "search_project_rag"):
            if name in available_names:
                mandatory_names.append(name)
    if role in {"coder", "coder_safe"} and is_implementation_intent(query):
        for name in _SOURCE_MUTATION_PRIORITY:
            if name in available_names:
                mandatory_names.append(name)
                break

    rows: list[dict[str, Any]] = []
    for index, schema in enumerate(tools):
        name = _tool_name(schema)
        document = _tool_document(schema)
        doc_tokens = _tokens(document)
        name_tokens = _tokens(name.replace("_", " "))
        lexical = 0.0
        if query_tokens and doc_tokens:
            lexical += len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        if query_tokens and name_tokens:
            lexical += 2.5 * len(query_tokens & name_tokens) / max(1, len(name_tokens))
        rows.append(
            {
                "index": index,
                "schema": schema,
                "name": name,
                "document": document,
                "score": lexical,
            }
        )

    shortlist_size = min(len(rows), max(top_k * 3, 12))
    lexical_ranked = sorted(rows, key=lambda item: (-float(item["score"]), item["index"]))
    shortlisted = lexical_ranked[:shortlist_size]
    by_name = {str(item["name"]): item for item in rows}
    for name in mandatory_names:
        item = by_name.get(name)
        if item is not None and item not in shortlisted:
            shortlisted.append(item)

    if query.strip() and shortlisted:
        try:
            reranked = router.rerank(
                query,
                [str(item["document"]) for item in shortlisted],
            )
            if len(reranked) == len(shortlisted):
                for item, score in zip(shortlisted, reranked):
                    item["score"] = float(item["score"]) + 4.0 * float(score)
        except Exception as exc:
            print(
                "small-model tool retrieval: reranker fallback",
                f"{type(exc).__name__}: {str(exc)[:240]}",
                flush=True,
            )

    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for name in mandatory_names:
        item = by_name.get(name)
        if item is not None and name not in selected_names:
            selected.append(item)
            selected_names.add(name)

    for item in sorted(shortlisted, key=lambda value: (-float(value["score"]), value["index"])):
        name = str(item["name"])
        if not name or name in selected_names:
            continue
        selected.append(item)
        selected_names.add(name)
        if len(selected) >= top_k:
            break

    external = {"external_mcp_capabilities", "external_mcp_schema", "external_mcp_call"}
    if selected_names & external:
        for name in ("external_mcp_schema", "external_mcp_call", "external_mcp_capabilities"):
            item = by_name.get(name)
            if item is None or name in selected_names:
                continue
            if len(selected) >= top_k:
                removable = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if str(selected[index]["name"]) not in mandatory_names
                        and str(selected[index]["name"]) not in external
                    ),
                    None,
                )
                if removable is None:
                    break
                selected_names.discard(str(selected[removable]["name"]))
                selected.pop(removable)
            selected.append(item)
            selected_names.add(name)

    selected.sort(key=lambda item: int(item["index"]))
    return tuple(item["schema"] for item in selected[:top_k])


def _install_tool_retrieval(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    current = cls._prepare_generation_request
    if getattr(current, "_mmm_small_model_tool_retrieval", False):
        return

    @wraps(current)
    def prepare_with_retrieved_tools(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ):
        stage, runtime, tools, request = current(self, role, messages, **kwargs)
        if not tools:
            return stage, runtime, tools, request

        selected = select_tool_schemas(
            self,
            role=role,
            query=_request_query(messages),
            tool_schemas=tools,
            require_fresh_evidence=bool(getattr(self, "_agent_require_fresh_evidence", False)),
        )
        if selected == tuple(tools):
            return stage, runtime, tools, request

        from .agent_capability_context import build_agent_capability_context

        cleaned_messages = [
            dict(message)
            for message in request.messages
            if not (
                str(message.get("role", "")) == "system"
                and isinstance(message.get("content"), str)
                and str(message.get("content", "")).startswith(_CAPABILITY_PREFIX)
            )
        ]
        rebuilt_messages = model_router_module._inject_system_context(
            cleaned_messages,
            build_agent_capability_context(stage, selected, model_role=role),
        )
        rebuilt = replace(
            request,
            messages=tuple(rebuilt_messages),
            tools=selected,
            tool_validation_schemas=selected,
            tool_choice="auto" if selected else None,
            parallel_tool_calls=True if selected else False,
        )
        print(
            "small-model tool retrieval:",
            f"role={role}",
            f"stage={stage}",
            f"eligible={len(tools)}",
            f"selected={len(selected)}",
            flush=True,
        )
        return stage, runtime, selected, rebuilt

    prepare_with_retrieved_tools._mmm_small_model_tool_retrieval = True  # type: ignore[attr-defined]
    prepare_with_retrieved_tools.__wrapped__ = current  # type: ignore[attr-defined]
    cls._prepare_generation_request = prepare_with_retrieved_tools


def _install_code_rag(pre_design_module: Any, production_tools_module: Any) -> None:
    """Turn the existing code RAG into hybrid lexical/semantic/reranked retrieval."""

    forced = pre_design_module._forced_rag_bundle
    if not getattr(forced, "_mmm_small_model_router_context", False):
        @wraps(forced)
        def forced_with_router(router: Any, research_brief: Mapping[str, Any]):
            token = _RAG_ROUTER.set(router)
            try:
                return forced(router, research_brief)
            finally:
                _RAG_ROUTER.reset(token)

        forced_with_router._mmm_small_model_router_context = True  # type: ignore[attr-defined]
        forced_with_router.__wrapped__ = forced  # type: ignore[attr-defined]
        pre_design_module._forced_rag_bundle = forced_with_router

    search = pre_design_module._search_code_index
    if not getattr(search, "_mmm_small_model_hybrid_code_rag", False):
        @wraps(search)
        def search_hybrid(index_path: Path | None, query: str) -> dict[str, Any]:
            if index_path is None:
                return search(index_path, query)
            router = _RAG_ROUTER.get()
            if router is None:
                return search(index_path, query)

            from .rag_index import ProjectRAGIndex

            modes = ((True, True, "semantic+rerank"), (False, True, "lexical+rerank"))
            errors: list[str] = []
            for semantic, rerank, mode in modes:
                try:
                    result = ProjectRAGIndex(index_path).search_with_receipt(
                        query,
                        limit=8,
                        router=router,
                        semantic=semantic,
                        rerank=rerank,
                    )
                    return {
                        "schema_version": "mmm/forced-code-rag-query-v2",
                        "status": "searched",
                        "retrieval_mode": mode,
                        "hits": [asdict(hit) for hit in result.hits],
                        "receipt": asdict(result.receipt),
                        "fallback_errors": errors,
                    }
                except Exception as exc:
                    errors.append(f"{mode}:{type(exc).__name__}:{str(exc)[:480]}")
            fallback = search(index_path, query)
            if isinstance(fallback, dict):
                fallback = dict(fallback)
                fallback["retrieval_mode"] = "lexical-fallback"
                fallback["fallback_errors"] = errors
            return fallback

        search_hybrid._mmm_small_model_hybrid_code_rag = True  # type: ignore[attr-defined]
        search_hybrid.__wrapped__ = search  # type: ignore[attr-defined]
        pre_design_module._search_code_index = search_hybrid

    cls = production_tools_module.ProductionToolService
    index_project = cls.index_project_rag
    if not getattr(index_project, "_mmm_small_model_semantic_repair_index", False):
        @wraps(index_project)
        def index_with_semantics(
            self: Any,
            roots: Sequence[str],
            *,
            index_path: str = "rag/project-index.json",
            metadata: dict[str, Any],
            semantic: bool = False,
        ):
            repair_like = bool(metadata.get("source_commit")) and str(metadata.get("license", "")) == "project-local"
            return index_project(
                self,
                roots,
                index_path=index_path,
                metadata=metadata,
                semantic=True if repair_like else semantic,
            )

        index_with_semantics._mmm_small_model_semantic_repair_index = True  # type: ignore[attr-defined]
        index_with_semantics.__wrapped__ = index_project  # type: ignore[attr-defined]
        cls.index_project_rag = index_with_semantics


def _exact_fact_ledger(evidence: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Extract exact machine facts that must survive any context compaction."""

    diagnostics: list[dict[str, Any]] = []
    raw_diagnostics = evidence.get("diagnostics")
    if isinstance(raw_diagnostics, Mapping):
        values = raw_diagnostics.get("diagnostics")
        if isinstance(values, list):
            for item in values[:64]:
                if not isinstance(item, Mapping):
                    continue
                diagnostics.append(
                    {
                        key: item.get(key)
                        for key in (
                            "path", "uri", "message", "code", "severity", "line",
                            "column", "range",
                        )
                        if item.get(key) is not None
                    }
                )

    rendered = json.dumps(
        {"evidence": evidence, "context": context},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    manifest = context.get("manifest") if isinstance(context, Mapping) else None
    build = evidence.get("build") if isinstance(evidence, Mapping) else None
    return {
        "schema_version": "mmm/lossless-small-agent-ledger-v1",
        "paths": sorted(set(_PATH.findall(rendered)))[:256],
        "sha256": sorted({value.casefold() for value in _SHA.findall(rendered)})[:256],
        "resource_ids": sorted(set(_RESOURCE_ID.findall(rendered)))[:256],
        "versions": sorted(set(_VERSION.findall(rendered)))[:128],
        "diagnostics": diagnostics,
        "build": {
            key: build.get(key)
            for key in ("status", "error")
            if isinstance(build, Mapping) and build.get(key) is not None
        },
        "manifest": dict(manifest) if isinstance(manifest, Mapping) else {},
    }


def _failure_memory_path(root: Path) -> Path:
    return root / ".minecraft_ai" / "repair-failure-experience.jsonl"


def _regular_memory_path(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)


def _failure_row(
    self: Any,
    evidence: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    verifier: Mapping[str, Any],
) -> dict[str, Any]:
    signature = self._signature(dict(evidence))
    pattern: list[dict[str, Any]] = []
    for item in operations[:16]:
        pattern.append(
            {
                "operation": str(item.get("operation", "")),
                "path": str(item.get("path", "")),
                "content_sha256": (
                    "sha256:" + hashlib.sha256(str(item.get("content", "")).encode("utf-8")).hexdigest()
                    if item.get("content") is not None
                    else ""
                ),
            }
        )
    body: dict[str, Any] = {
        "schema_version": "mmm/verified-repair-failure-v1",
        "outcome": "FAIL",
        "signature": signature,
        "signature_sha256": "sha256:" + hashlib.sha256(signature.encode("utf-8")).hexdigest(),
        "verifier": dict(verifier),
        "repair_pattern": pattern,
        "exact_fact_ledger": _exact_fact_ledger(evidence, {}),
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    body["experience_id"] = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return body


def _record_verified_failure(root: Path, row: Mapping[str, Any]) -> None:
    path = _failure_memory_path(root)
    with _FAILURE_LOCK:
        if path.exists() and not _regular_memory_path(path):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = str(row.get("experience_id", ""))
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, Mapping) and str(value.get("experience_id", "")) == identity:
                            return
            except OSError:
                return
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            return


def _read_verified_failures(root: Path, signature: str, *, limit: int = 4) -> list[dict[str, Any]]:
    path = _failure_memory_path(root)
    if not path.is_file() or not _regular_memory_path(path):
        return []
    target = _tokens(signature)
    ranked: list[tuple[int, float, str, dict[str, Any]]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("outcome") != "FAIL":
                    continue
                source = str(row.get("signature", ""))
                exact = int(source == signature)
                values = _tokens(source)
                similarity = len(target & values) / max(1, len(target | values)) if target and values else 0.0
                ranked.append((exact, similarity, str(row.get("experience_id", "")), row))
    except OSError:
        return []
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        {
            "exact_signature": bool(exact),
            "similarity": round(similarity, 6),
            "verifier": row.get("verifier", {}),
            "repair_pattern": row.get("repair_pattern", []),
            "exact_fact_ledger": row.get("exact_fact_ledger", {}),
        }
        for exact, similarity, _identity, row in ranked[:limit]
        if exact or similarity > 0.0
    ]


def _install_repair_context(repair_module: Any, optimization_module: Any) -> None:
    """Add lossless facts and verified negative experience to fresh repair turns."""

    cls = repair_module.RepairEngine
    request_patch = cls._request_patch
    if not getattr(request_patch, "_mmm_small_model_isolated_repair", False):
        @wraps(request_patch)
        def request_with_isolated_context(self: Any, evidence: dict[str, Any], context: dict[str, Any]):
            root_value = getattr(self, "_mmm_agentic_root", None)
            if root_value:
                root = Path(root_value).expanduser().resolve()
            else:
                active = repair_module._ACTIVE_REPAIR_PROJECT_INDEX.get()
                root = active[0] if active is not None else None
            signature = self._signature(evidence)
            failures = _read_verified_failures(root, signature) if root is not None else []
            fresh_context = dict(context)
            fresh_context["exact_fact_ledger"] = _exact_fact_ledger(evidence, context)
            fresh_context["fresh_context_contract"] = {
                "fresh_executor": True,
                "prior_model_reasoning_inherited": False,
                "compaction_boundary": "repair_attempt",
                "lossless_fields_authoritative": True,
            }
            if failures:
                fresh_context["verified_failure_memory"] = {
                    "policy": (
                        "Host-verifier-proven failed repair patterns. Do not repeat an exact "
                        "failed patch shape unless new machine evidence invalidates the old result."
                    ),
                    "matches": failures,
                }
            return request_patch(self, evidence, fresh_context)

        request_with_isolated_context._mmm_small_model_isolated_repair = True  # type: ignore[attr-defined]
        request_with_isolated_context.__wrapped__ = request_patch  # type: ignore[attr-defined]
        cls._request_patch = request_with_isolated_context

    verify = optimization_module._verify_repair_candidate
    if not getattr(verify, "_mmm_verified_failure_memory", False):
        @wraps(verify)
        def verify_and_record(
            self: Any,
            root: Path | None,
            operations: Sequence[Mapping[str, Any]],
            evidence: Mapping[str, Any],
        ):
            score, verifier = verify(self, root, operations, evidence)
            try:
                error_count = int(verifier.get("jdt_error_count"))
            except (TypeError, ValueError):
                error_count = 0
            status = str(verifier.get("jdt_status", ""))
            if root is not None and error_count > 0 and status not in {"", "UNAVAILABLE", "VERIFIER_ERROR", "NOT_RUN"}:
                _record_verified_failure(
                    Path(root).resolve(),
                    _failure_row(self, evidence, operations, verifier),
                )
            return score, verifier

        verify_and_record._mmm_verified_failure_memory = True  # type: ignore[attr-defined]
        verify_and_record.__wrapped__ = verify  # type: ignore[attr-defined]
        optimization_module._verify_repair_candidate = verify_and_record


def install(
    *,
    model_router_module: Any,
    pre_design_rag_module: Any,
    production_tools_module: Any,
    repair_module: Any,
    optimization_module: Any,
) -> None:
    """Compose frozen-small-model inference amplifiers exactly once."""

    _install_tool_retrieval(model_router_module)
    _install_code_rag(pre_design_rag_module, production_tools_module)
    _install_repair_context(repair_module, optimization_module)


__all__ = ["_exact_fact_ledger", "_request_query", "install", "select_tool_schemas"]
