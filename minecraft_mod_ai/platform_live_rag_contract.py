from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

_RAG_THREAD_STATE = threading.local()


def _thread_index(retrieval: Any) -> Any:
    """Build the immutable builtin corpus index once per retrieval worker thread."""
    key = (id(retrieval.OfficialCorpusIndex), id(retrieval.BUILTIN_CORPUS))
    indexes = getattr(_RAG_THREAD_STATE, "indexes", None)
    if indexes is None:
        indexes = {}
        _RAG_THREAD_STATE.indexes = indexes
    index = indexes.get(key)
    if index is None:
        index = retrieval.OfficialCorpusIndex(documents=retrieval.BUILTIN_CORPUS)
        indexes[key] = index
    return index


def _replace_kwonly_default(function: Any, name: str, value: Any) -> None:
    defaults = getattr(function, "__kwdefaults__", None)
    if not isinstance(defaults, dict) or name not in defaults:
        return
    updated = dict(defaults)
    updated[name] = value
    function.__kwdefaults__ = updated


def _resolved_target(
    module: Any,
    minecraft_version: str | None,
    loader: str | None,
    mappings: str | None,
) -> tuple[str, str, str] | None:
    """Return a canonical target only when all supplied target data is executable.

    Live research is allowed to run without this value. Missing, partial or stale target
    metadata is therefore an optional refinement signal, never a retrieval precondition.
    """
    version = str(minecraft_version or "").strip()
    loader_id = str(loader or "").strip().casefold()
    mapping_id = str(mappings or "").strip()
    if not version or not loader_id or not mapping_id:
        return None
    try:
        adapter = module.adapter_for_target(version, loader_id)
    except Exception:
        return None
    if mapping_id != str(getattr(adapter, "yarn_mappings", "")).strip():
        return None
    return (
        str(getattr(adapter, "minecraft_version", version)).strip(),
        str(getattr(adapter, "loader", loader_id)).strip().casefold(),
        mapping_id,
    )


def _resolved_brief_target(central_module: Any, research_brief: Mapping[str, Any]) -> bool:
    raw_target = research_brief.get("_mmm_platform_target")
    if not isinstance(raw_target, Mapping):
        return False
    return _resolved_target(
        central_module,
        raw_target.get("minecraft_version") or raw_target.get("game_version"),
        raw_target.get("loader"),
        raw_target.get("mappings"),
    ) is not None


def _generic_retrieve(
    retrieval: Any,
    index: Any,
    query: str,
    *,
    limit: int,
) -> Any:
    """Retrieve target-neutral official evidence without fabricating a platform target."""
    query = str(query).strip()
    if not 2 <= len(query) <= 2_000:
        raise retrieval.SpecValidationError("RAG query length must be between 2 and 2000.")
    if type(limit) is not int or not 1 <= limit <= 12:
        raise retrieval.SpecValidationError("RAG result limit must be between 1 and 12.")

    family = retrieval._classify_query(query)
    canonical = retrieval._canonical_query(query, family)
    eligible = {document.document_id: document for document in index.documents}
    query_terms = frozenset(retrieval._tokens(canonical))
    query_grams = retrieval._trigrams(canonical)
    graph_boost: dict[str, float] = {document_id: 0.0 for document_id in eligible}
    lexical: dict[str, float] = {}
    semantic: dict[str, float] = {}
    family_score: dict[str, float] = {}

    for document_id, document in eligible.items():
        searchable = " ".join((document.title, document.content, *document.topics))
        document_terms = frozenset(retrieval._tokens(searchable))
        lexical[document_id] = len(query_terms & document_terms) / max(1, len(query_terms))
        semantic[document_id] = retrieval._jaccard(
            query_grams,
            retrieval._trigrams(searchable),
        )
        family_score[document_id] = 1.0 if family in document.families else 0.0

    lexical_order = sorted(
        eligible,
        key=lambda document_id: (-lexical[document_id], document_id),
    )
    for rank, document_id in enumerate(lexical_order[:5], start=1):
        graph_boost[document_id] += 1.0 / rank
        for related_id in eligible[document_id].related_ids:
            if related_id in graph_boost:
                graph_boost[related_id] += 0.45 / rank

    score: dict[str, float] = {}
    channels: dict[str, tuple[str, ...]] = {}
    for document_id in eligible:
        score[document_id] = (
            0.42 * lexical[document_id]
            + 0.28 * semantic[document_id]
            + 0.20 * family_score[document_id]
            + 0.10 * min(1.0, graph_boost[document_id])
        )
        active: list[str] = []
        if lexical[document_id] > 0:
            active.append("lexical")
        if semantic[document_id] > 0:
            active.append("semantic")
        if family_score[document_id] > 0:
            active.append("family")
        if graph_boost[document_id] > 0:
            active.append("graph")
        channels[document_id] = tuple(active)

    ordered = sorted(
        eligible,
        key=lambda document_id: (-score[document_id], document_id),
    )[:limit]
    hits: list[Any] = []
    for rank, document_id in enumerate(ordered, start=1):
        document = eligible[document_id]
        evidence_seed = retrieval.canonical_json(
            {
                "query": canonical,
                "document_id": document_id,
                "content_sha256": document.content_sha256,
                "rank": rank,
                "snapshot": index.snapshot_hash,
                "target": None,
            }
        ).encode("utf-8")
        hits.append(
            retrieval.RetrievalHit(
                evidence_id="sha256:" + hashlib.sha256(evidence_seed).hexdigest(),
                document_id=document_id,
                title=document.title,
                url=document.url,
                excerpt=document.content,
                content_sha256=document.content_sha256,
                revision=document.revision,
                minecraft_versions=("*",),
                score=round(score[document_id], 8),
                channels=channels[document_id],
            )
        )

    family_hits = sum(family in eligible[hit.document_id].families for hit in hits)
    signal_hits = sum(bool(hit.channels) for hit in hits)
    coverage = min(
        1.0,
        0.6 * family_hits / max(1, min(2, len(hits)))
        + 0.4 * signal_hits / max(1, min(3, len(hits))),
    )
    quality = (
        "strong"
        if hits
        and signal_hits >= min(2, len(hits))
        and (family_hits > 0 or family == "project")
        else "weak"
    )
    correction_required = quality != "strong"
    corrections = (
        (
            f"{family} official API concepts",
            f"{family} compatibility and mapping constraints",
            f"{family} deterministic runtime validation",
        )
        if correction_required
        else ()
    )
    query_hash = "sha256:" + hashlib.sha256(
        retrieval.canonical_json(
            {
                "query": query,
                "canonical": canonical,
                "family": family,
                "target": None,
            }
        ).encode("utf-8")
    ).hexdigest()
    return retrieval.RetrievalReceipt(
        schema_version="minecraft-mod-ai/retrieval-receipt-v1",
        query=query,
        canonical_query=canonical,
        query_family=family,
        minecraft_version="",
        loader="",
        mappings="",
        query_hash=query_hash,
        corpus_snapshot_hash=index.snapshot_hash,
        quality=quality,
        coverage=round(coverage, 6),
        correction_required=correction_required,
        correction_queries=corrections,
        hits=tuple(hits),
    )


def _build_generic_research_graph(
    central_module: Any,
    research_brief: dict[str, Any],
    retrieve: Callable[..., Any],
) -> dict[str, Any]:
    domains = research_brief.get("domains")
    if not isinstance(domains, list) or not domains:
        raise central_module.SpecValidationError("Central research brief has no domains.")

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for raw_domain in domains:
        domain = central_module._research_domain(raw_domain)
        if "official_docs" not in domain.providers:
            results.append(
                {
                    "domain_id": domain.domain_id,
                    "strategy": "routed_to_other_providers",
                    "queries": [],
                }
            )
            continue

        query_results: list[dict[str, Any]] = []
        has_hits = False
        for query in domain.queries:
            primary = retrieve(query, limit=8)
            corrections: list[dict[str, Any]] = []
            for correction_query in primary.correction_queries:
                correction = retrieve(correction_query, limit=4)
                corrections.append(correction.to_dict())
                has_hits = has_hits or bool(correction.hits)
            has_hits = has_hits or bool(primary.hits)
            query_results.append(
                {
                    "query_sha256": central_module._sha256(query),
                    "strategy": (
                        "single"
                        if not primary.correction_required
                        else "corrective_multi_hop"
                    ),
                    "primary": primary.to_dict(),
                    "corrections": corrections,
                }
            )
        if not has_hits:
            unresolved.append(domain.domain_id)
        results.append(
            {
                "domain_id": domain.domain_id,
                "strategy": "adaptive_generic_per_query",
                "queries": query_results,
            }
        )

    payload = {
        "schema_version": "mmm/central-evidence-graph-v1",
        "brief_sha256": research_brief.get("brief_sha256", ""),
        "target": None,
        "domains": results,
        "deferred_official_domains": [],
        "unresolved_official_domains": unresolved,
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = central_module._sha256(
        central_module.canonical_json(payload)
    )
    return payload


def _install_optional_central_boundary(
    central_module: Any,
    shared_retrieve: Callable[..., Any],
) -> None:
    current_build = central_module._build_research_graph
    if not getattr(current_build, "_mmm_optional_platform_target", False):

        @wraps(current_build)
        def optional_build(
            research_brief: dict[str, Any],
            *,
            retrieve: Callable[..., Any] | None = None,
        ) -> dict[str, Any]:
            selected_retrieve = retrieve or shared_retrieve
            if _resolved_brief_target(central_module, research_brief):
                return current_build(research_brief, retrieve=selected_retrieve)
            return _build_generic_research_graph(
                central_module,
                research_brief,
                selected_retrieve,
            )

        optional_build._mmm_optional_platform_target = True
        central_module._build_research_graph = optional_build
        central_module._serial_retrieve_domain_evidence = optional_build

    from . import parallel_runtime_contract as parallel_module

    current_parallel = parallel_module.retrieve_domain_evidence
    if not getattr(current_parallel, "_mmm_optional_platform_target", False):

        @wraps(current_parallel)
        def optional_parallel(
            research_brief: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if isinstance(research_brief, Mapping) and not _resolved_brief_target(
                central_module,
                research_brief,
            ):
                selected_retrieve = kwargs.get("retrieve") or shared_retrieve
                return central_module._build_research_graph(
                    research_brief,
                    retrieve=selected_retrieve,
                )
            return current_parallel(research_brief, *args, **kwargs)

        optional_parallel._mmm_optional_platform_target = True
        parallel_module.retrieve_domain_evidence = optional_parallel


def install(*, retrieval_module: Any) -> None:
    """Install target-optional live retrieval.

    A valid platform target narrows retrieval exactly as before. Missing, partial or stale
    target metadata falls back to target-neutral corpus retrieval and never aborts research.
    """
    cls = retrieval_module.OfficialCorpusIndex
    original = cls.retrieve
    if not getattr(original, "_mmm_live_platform_rag", False):

        @wraps(original)
        def retrieve(
            self: Any,
            query: str,
            *,
            minecraft_version: str | None = None,
            loader: str | None = None,
            mappings: str | None = None,
            limit: int = 6,
        ):
            target = _resolved_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
            )
            if target is None:
                return _generic_retrieve(
                    retrieval_module,
                    self,
                    query,
                    limit=limit,
                )
            version, loader_id, mapping_id = target
            return original(
                self,
                query,
                minecraft_version=version,
                loader=loader_id,
                mappings=mapping_id,
                limit=limit,
            )

        retrieve._mmm_live_platform_rag = True
        cls.retrieve = retrieve

    current_public_retrieve = retrieval_module.retrieve_official_evidence
    if getattr(current_public_retrieve, "_mmm_thread_local_index_reuse", False):
        shared_retrieve = current_public_retrieve
    else:

        @wraps(current_public_retrieve)
        def shared_retrieve(
            query: str,
            *,
            minecraft_version: str | None = None,
            loader: str | None = None,
            mappings: str | None = None,
            limit: int = 6,
        ):
            return _thread_index(retrieval_module).retrieve(
                query,
                minecraft_version=minecraft_version,
                loader=loader,
                mappings=mappings,
                limit=limit,
            )

        shared_retrieve._mmm_thread_local_index_reuse = True
        retrieval_module.retrieve_official_evidence = shared_retrieve

    from . import central_research as central_module

    central_module.retrieve_official_evidence = shared_retrieve
    _replace_kwonly_default(
        central_module.retrieve_domain_evidence,
        "retrieve",
        shared_retrieve,
    )
    _install_optional_central_boundary(central_module, shared_retrieve)


__all__ = ["install"]
