from __future__ import annotations

import re
from functools import wraps
from typing import Any

_FINE_ROUTE_MARKER = "__mmm_research_fine_grained_code_route_v1__"
_VERSION = re.compile(r"(?<![0-9])(?:1\.)?[0-9]{1,2}(?:\.[0-9]{1,3}){1,2}(?![0-9])")
_TRACE = (
    "traceback", "stack trace", "exception", "cannot find", "cannot resolve",
    "diagnostic", "compile error", "compilation failed", "gradle failed", "gametest failed",
    "jdt", "no such method", "no such field", "symbol not found",
)
_RIPPLE = (
    "rename", "refactor", "impact", "ripple", "all usages", "all callers",
    "all references", "affected callers", "affected dependencies",
)
_API = (
    " api ", "signature", "callback", "codec", "registry", "register", "mapping namespace",
    "method contract", "interface contract", "event hook", "packet handler", "payload type",
)
_PROCEDURAL = (
    "implement", "implementation flow", "create", "generate", "load", "save", "persist",
    "serialize", "deserialize", "sync", "spawn", "initialize", "bootstrap", "update flow",
)


def harden_code_search_routes() -> None:
    """Add research-level fine routes while reusing the existing hybrid search engine.

    The installed hybrid engine already owns lexical/semantic/rerank/relation modes,
    retry, centroid adaptation and snapshot caching. This wrapper only classifies
    evidence need more precisely and steers that existing portfolio; it never creates
    a second index or treats retrieved examples as authority.
    """
    from .production_tools import ProductionToolService

    current = ProductionToolService.search_code_rag
    if getattr(current, _FINE_ROUTE_MARKER, False):
        return

    @wraps(current)
    def search_code_rag(
        self: Any,
        query: str,
        *,
        index_path: str = "rag/project-index.json",
        limit: int = 8,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        original = str(query or "").strip()
        fine_route = classify_code_evidence_need(original)
        routed = route_query(original, fine_route)
        result = dict(
            current(
                self,
                routed,
                index_path=index_path,
                limit=limit,
                semantic=semantic,
                rerank=rerank,
                required_metadata=required_metadata,
            )
        )
        # Preserve the user's exact question as the public query while retaining
        # the steering query and the pre-existing coarse route for auditability.
        result["query"] = original
        result["fine_task_route"] = fine_route
        result["fine_route_query"] = routed
        result["coarse_task_route"] = result.get("task_route")
        result["portfolio_policy"] = portfolio_policy(fine_route)
        result["retrieved_context_authority"] = "evidence_only_current_host_contract_wins"
        return result

    setattr(search_code_rag, _FINE_ROUTE_MARKER, True)
    ProductionToolService.search_code_rag = search_code_rag


def classify_code_evidence_need(query: str) -> str:
    value = f" {str(query or '').casefold()} "
    if _VERSION.search(value) and any(
        marker in value for marker in ("minecraft", "fabric", "forge", "mapping", "yarn", "version")
    ):
        return "exact_version"
    if any(marker in value for marker in _TRACE):
        return "trace"
    if any(marker in value for marker in _RIPPLE):
        return "ripple"
    if any(marker in value for marker in _API):
        return "api"
    if any(marker in value for marker in _PROCEDURAL):
        return "procedural"
    if any(
        marker in value
        for marker in ("dependency", "depends", "caller", "callee", "import", "extends", "implements")
    ):
        return "dependency"
    if any(marker in value for marker in ("whole project", "entire project", "architecture", "overview")):
        return "global"
    return "exact_or_semantic"


def route_query(query: str, route: str) -> str:
    """Steer the already-installed hybrid portfolio without inventing a new backend."""
    suffix = {
        "trace": "diagnostic failing symbol dependency call chain imports callers callees exact declaration usage",
        "ripple": "dependency call chain callers callees imports extends implements all references affected usages",
        "api": "api method interface callback codec registry declaration usage implementation contract",
        "procedural": "implementation flow class method resource load save register sync validate behavior",
        "dependency": "dependency call chain imports callers callees",
        "global": "entire project architecture related modules dependencies",
    }.get(route, "")
    return query if not suffix else f"{query}\n{suffix}"


def portfolio_policy(route: str) -> dict[str, Any]:
    if route in {"trace", "ripple", "dependency"}:
        order = ["lexical+relations", "rerank+relations", "semantic+rerank+relations"]
    elif route == "api":
        order = ["exact-symbol-lexical", "lexical+rerank", "semantic+rerank"]
    elif route == "procedural":
        order = ["semantic+rerank", "lexical+rerank", "procedure-alignment-in-host-grounding"]
    elif route == "exact_version":
        order = ["exact-lexical", "lexical+rerank", "semantic+rerank-fallback"]
    elif route == "global":
        order = ["lexical+global-relations", "semantic+rerank+global-relations", "centroid-adaptation"]
    else:
        order = ["selective-exact-or-semantic", "rerank-when-available"]
    return {
        "family_order": order,
        "selective_retrieval": True,
        "generic_similar_code_authoritative": False,
        "current_exact_source_authoritative": True,
    }


__all__ = [
    "classify_code_evidence_need",
    "harden_code_search_routes",
    "portfolio_policy",
    "route_query",
]
