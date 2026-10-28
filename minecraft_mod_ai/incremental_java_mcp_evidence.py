"""Host-owned, read-only Minecraft MCP evidence on a local API lookup miss.

The model remains on strict schema-constrained pages (tools=()). External provider
data is a *reference*, never an authority to create a Java symbol, choose a
PlatformLock target, mutate workspace files, or change the tool surface.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


_CAPABILITIES = ("source_search", "official_mod_docs")
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


def augment_api_lookup(builder: Any, query: str, local_page: Mapping[str, Any]) -> dict[str, Any]:
    """Perform one cached host-side evidence query when verified local symbols miss.

    On failure, preserve exactly the ordinary local lookup response. Only an
    already bound production project or an explicit test router can use MCP.
    """
    page = dict(local_page)
    question = str(query or "").strip()
    if (not question or question.startswith("@") or page.get("total") != 0
            or page.get("offset", 0) != 0):
        return page
    context = builder.context
    if os.environ.get("MMM_HOST_MCP_EVIDENCE", "1").strip().lower() in {"0", "false", "off"}:
        return page
    if not context.get("project_root") and "host_mcp_router" not in context:
        return page
    target = context.get("target")
    if not isinstance(target, Mapping) or not str(target.get("minecraft_version") or "").strip():
        return page
    cache = context.setdefault("_incremental_mcp_evidence_cache", {})
    key = json.dumps({"query": question, "target": dict(target)}, ensure_ascii=False,
                     sort_keys=True, default=str)
    if key in cache:
        answer = cache[key]
        if answer is not None:
            page["external_mcp_evidence"] = dict(answer)
        return page

    router = context.get("host_mcp_router")
    if router is None:
        from .external_mcp_router import ExternalMCPRouter
        router = ExternalMCPRouter(timeout_seconds=30.0)
        context["host_mcp_router"] = router

    # Prefer a precise source search for qualified Java API names; documentation
    # search is less expensive and better suited to natural-language queries.
    candidates = (_CAPABILITIES if "." in question or "#" in question
                  else tuple(reversed(_CAPABILITIES)))
    answer = None
    for capability in candidates:
        try:
            bundle = router.invoke(
                capability, stage="generation", max_access="read",
                target=target, arguments={"query": question}, corroborate=1,
                required=False)
        except (RuntimeError, ValueError, OSError, TypeError):
            continue
        if not isinstance(bundle, Mapping) or bundle.get("status") != "PASS":
            continue
        for receipt in bundle.get("evidence") or ():
            if not isinstance(receipt, Mapping):
                continue
            snippet = _snippet(receipt)
            if not snippet:
                continue
            answer = {
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
            break
        if answer is not None:
            break
    cache[key] = answer  # Cache misses too: never repeat the same failing I/O.
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
