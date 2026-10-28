"""Host-owned, read-only Minecraft MCP evidence on a local API lookup miss.

The model remains on strict schema-constrained pages (tools=()). External provider
data is a *reference*, never an authority to create a Java symbol, choose a
PlatformLock target, mutate workspace files, or change the tool surface.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any


_CAPABILITIES = ("official_mod_docs", "source_search")
_MAPPING_TERMS = ("mapping", "yarn", "mojmap", "obfuscat", "remap")
_REGISTRY_TERMS = ("registry", "registries", "registrar")
_EXAMPLE_TERMS = ("example", "sample mod", "tutorial")
_SNIPPET_CHARS = 3600


def _snippet(receipt: Mapping[str, Any]) -> str:
    result = receipt.get("result")
    if not isinstance(result, Mapping):
        return ""
    # Raw provider content is never instructions to the host. Keep a small,
    # attributable extract; do not copy provider metadata or arbitrary schemas.
    for key in ("structured", "parsed_text", "text"):
        value = result.get(key)
        if value in (None, "", [], {}):
            continue
        raw = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str)
        if raw.strip():
            return raw[:_SNIPPET_CHARS]
    return ""


def _eligible(builder: Any, query: str, page: Mapping[str, Any]) -> bool:
    context = builder.context
    target = context.get("target")
    return bool(
        query and not query.startswith("@")
        and page.get("total") == 0 and page.get("offset", 0) == 0
        and os.environ.get("MMM_HOST_MCP_EVIDENCE", "1").strip().lower()
        not in {"0", "false", "off"}
        and (context.get("project_root") or "host_mcp_router" in context)
        and isinstance(target, Mapping)
        and str(target.get("minecraft_version") or "").strip()
    )


def _record(context: dict[str, Any], event: str) -> None:
    counts = context.setdefault("_incremental_mcp_evidence_metrics", {})
    counts[event] = counts.get(event, 0) + 1


def _capabilities_for_query(question: str) -> tuple[str, ...]:
    """Cheap deterministic routing, without making a 9B model pick MCP tools."""
    lowered = question.casefold()
    if any(token in lowered for token in _MAPPING_TERMS):
        return ("mapping_resolution", *_CAPABILITIES)
    if "." not in question and "#" not in question:
        if any(token in lowered for token in _REGISTRY_TERMS):
            return ("registry_lookup", *_CAPABILITIES)
        if any(token in lowered for token in _EXAMPLE_TERMS):
            return ("mod_examples", *_CAPABILITIES)
    if "#" in question:
        return ("source_search", "official_mod_docs")
    # Prefer faster documentation over starting Minecraft source decompilation
    # for an ordinary type/behavior query.
    return _CAPABILITIES


def _search_external(router: Any, question: str, target: Mapping[str, Any],
                     context: dict[str, Any]) -> dict[str, str] | None:
    candidates = _capabilities_for_query(question)
    for capability in candidates:
        _record(context, "provider_attempts")
        try:
            bundle = router.invoke(
                capability, stage="generation", max_access="read",
                target=target, arguments={"query": question}, corroborate=1,
                required=False)
        except Exception:  # noqa: BLE001 - optional provider never fails code authoring
            _record(context, "provider_errors")
            continue
        if not isinstance(bundle, Mapping) or bundle.get("status") != "PASS":
            continue
        for receipt in bundle.get("evidence") or ():
            if not isinstance(receipt, Mapping):
                continue
            snippet = _snippet(receipt)
            if not snippet:
                continue
            _record(context, "provider_successes")
            return {
                "query": question,
                "status": "UNVERIFIED_EXTERNAL_REFERENCE",
                "capability": capability,
                "provider": str(receipt.get("server") or ""),
                "tool": str(receipt.get("tool") or ""),
                "receipt_sha256": str(receipt.get("result_sha256") or ""),
                "snippet": snippet,
                "usage": "Reference only. Not a verified callable Java symbol. "
                         "Validate signatures against the bound target or compiler.",
            }
    return None


def augment_api_lookup(builder: Any, query: str, local_page: Mapping[str, Any]) -> dict[str, Any]:
    """Add optional cached evidence without changing authoritative Java symbols."""
    page = dict(local_page)
    question = str(query or "").strip()
    if not _eligible(builder, question, page):
        return page
    context = builder.context
    target = context["target"]
    cache = context.setdefault("_incremental_mcp_evidence_cache", {})
    key = json.dumps({"query": question, "target": dict(target)}, ensure_ascii=False,
                     sort_keys=True, default=str)
    if key in cache:
        _record(context, "cache_hits")
        answer = cache[key]
    else:
        router = context.get("host_mcp_router")
        if router is None:
            from .external_mcp_router import ExternalMCPRouter
            router = ExternalMCPRouter(timeout_seconds=30.0, reuse_sessions=True)
            context["_host_mcp_router_managed"] = True
            context["host_mcp_router"] = router
        _record(context, "cache_misses")
        started = time.perf_counter()
        try:
            answer = _search_external(router, question, target, context)
        finally:
            metrics = context.setdefault("_incremental_mcp_evidence_metrics", {})
            metrics["remote_seconds"] = round(
                metrics.get("remote_seconds", 0.0) + time.perf_counter() - started, 3
            )
        cache[key] = answer  # A failed lookup is also a useful cache entry.
    if answer is not None:
        page["external_mcp_evidence"] = dict(answer)
    return page


def recover_compiler_api_evidence(builder: Any, diagnostic: str) -> None:
    """Retrieve target-scoped reference data for an external API compiler error.

    The host, not MiMo, chooses the evidence route. Project-local type/variable
    mistakes must go straight to operand repair, without network or MCP work.
    """
    from .generation_evidence_controller import repair_evidence_route_for_errors

    errors = [{"message": str(diagnostic)}]
    route = repair_evidence_route_for_errors(
        errors, target_path=builder.spec.get("target_path"))
    if route.get("route") != "official_api":
        return
    query = str(route.get("query_prefix") or "").strip()
    if not query:
        return
    local_page = builder.api.lookup(query, 0)
    builder.latest_api = augment_api_lookup(builder, query, local_page)



from contextlib import contextmanager


@contextmanager
def coding_mcp_scope(context):
    """Dispose warm MCP transports when a construction file finishes or fails."""
    try:
        yield
    finally:
        if context.pop("_host_mcp_router_managed", False):
            router = context.pop("host_mcp_router", None)
            if router is not None:
                try:
                    router.close()
                except Exception:  # noqa: BLE001 - cleanup cannot hide a build diagnostic
                    _record(context, "session_close_errors")
