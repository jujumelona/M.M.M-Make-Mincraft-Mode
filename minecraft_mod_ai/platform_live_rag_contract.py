from __future__ import annotations

import hashlib
import threading
from functools import wraps
from typing import Any


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


def _required_target(
    retrieval: Any,
    minecraft_version: str | None,
    loader: str | None,
    mappings: str | None,
) -> tuple[str, str, str]:
    version = str(minecraft_version or "").strip()
    loader_id = str(loader or "").strip().casefold()
    mapping_id = str(mappings or "").strip()
    if not version or not loader_id or not mapping_id:
        raise retrieval.SpecValidationError(
            "Official RAG requires a host-selected minecraft_version, loader and mappings."
        )
    return version, loader_id, mapping_id


def install(*, retrieval_module: Any) -> None:
    """Install one target-required retrieval path for every executable target."""

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
            version, loader_id, mapping_id = _required_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
            )
            return _retrieve_live(
                retrieval_module,
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
            version, loader_id, mapping_id = _required_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
            )
            return _thread_index(retrieval_module).retrieve(
                query,
                minecraft_version=version,
                loader=loader_id,
                mappings=mapping_id,
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


def _retrieve_live(
    retrieval: Any,
    index: Any,
    query: str,
    *,
    minecraft_version: str,
    loader: str,
    mappings: str,
    limit: int,
):
    query = str(query).strip()
    if not 2 <= len(query) <= 2_000:
        raise retrieval.SpecValidationError("RAG query length must be between 2 and 2000.")
    if loader != "fabric":
        raise retrieval.SpecValidationError(
            f"No reviewed local official corpus is installed for loader={loader!r}."
        )
    if not minecraft_version or not mappings:
        raise retrieval.SpecValidationError("Live RAG requires an approved target and mappings policy.")
    if type(limit) is not int or not 1 <= limit <= 12:
        raise retrieval.SpecValidationError("RAG result limit must be between 1 and 12.")

    family = retrieval._classify_query(query)
    canonical = retrieval._canonical_query(query, family)
    eligible = {
        document.document_id: document
        for document in index.documents
        if (
            minecraft_version in document.minecraft_versions
            or "*" in document.minecraft_versions
        )
        and document.loader in {loader, "agnostic"}
        and not (
            document.minecraft_versions != ("*",)
            and minecraft_version not in document.minecraft_versions
        )
    }
    if not eligible:
        raise retrieval.SpecValidationError(
            f"No official corpus documents apply to Minecraft {minecraft_version}/{loader}."
        )

    lexical = index._lexical_ranking(canonical, eligible)
    semantic = index._semantic_ranking(canonical, eligible)
    graph = index._graph_ranking(lexical, eligible)
    family_rank = sorted(
        eligible,
        key=lambda document_id: (
            family not in eligible[document_id].families,
            document_id,
        ),
    )
    rankings = {
        "bm25": lexical,
        "semantic_fallback": semantic,
        "graph": graph,
        "family_filter": family_rank,
    }
    weights = {
        "bm25": 1.0,
        "semantic_fallback": 0.55,
        "graph": 0.35,
        "family_filter": 0.7,
    }
    scores = {document_id: 0.0 for document_id in eligible}
    channels = {document_id: [] for document_id in eligible}
    for channel, ranking in rankings.items():
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] += weights[channel] / (60 + rank)
            if rank <= max(limit * 2, 8):
                channels[document_id].append(channel)
    ordered = sorted(
        eligible,
        key=lambda document_id: (-scores[document_id], document_id),
    )[:limit]

    hits = []
    for rank, document_id in enumerate(ordered, start=1):
        document = eligible[document_id]
        evidence_seed = retrieval.canonical_json(
            {
                "query": canonical,
                "document_id": document_id,
                "content_sha256": document.content_sha256,
                "rank": rank,
                "snapshot": index.snapshot_hash,
                "target": minecraft_version,
                "loader": loader,
                "mappings": mappings,
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
                minecraft_versions=document.minecraft_versions,
                score=round(scores[document_id], 8),
                channels=tuple(channels[document_id]),
            )
        )

    family_hits = sum(family in eligible[hit.document_id].families for hit in hits)
    exact_hits = sum(minecraft_version in hit.minecraft_versions for hit in hits)
    applicable_hits = sum(
        minecraft_version in hit.minecraft_versions or "*" in hit.minecraft_versions
        for hit in hits
    )
    coverage = min(
        1.0,
        (applicable_hits / max(1, min(3, len(hits)))) * 0.55
        + (family_hits / max(1, min(3, len(hits)))) * 0.45,
    )
    quality = "strong" if len(hits) >= min(3, limit) and family_hits >= 1 else "weak"
    correction_required = quality != "strong"
    corrections = (
        (
            f"{family} official {loader} documentation Minecraft {minecraft_version}",
            f"{family} {mappings} current symbol signature",
            f"{family} compile runtime validation",
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
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
                "exact_hits": exact_hits,
            }
        ).encode("utf-8")
    ).hexdigest()
    return retrieval.RetrievalReceipt(
        schema_version="minecraft-mod-ai/retrieval-receipt-v1",
        query=query,
        canonical_query=canonical,
        query_family=family,
        minecraft_version=minecraft_version,
        loader=loader,
        mappings=mappings,
        query_hash=query_hash,
        corpus_snapshot_hash=index.snapshot_hash,
        quality=quality,
        coverage=round(coverage, 6),
        correction_required=correction_required,
        correction_queries=corrections,
        hits=tuple(hits),
    )


__all__ = ["install"]
