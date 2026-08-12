from __future__ import annotations

import hashlib
from functools import wraps
from typing import Any


def install(*, retrieval_module: Any, platform_planning_module: Any) -> None:
    """Use only version-applicable or version-neutral documents for live targets."""

    cls = retrieval_module.OfficialCorpusIndex
    original = cls.retrieve
    if not getattr(original, "_mmm_live_platform_rag", False):

        @wraps(original)
        def retrieve(
            self: Any,
            query: str,
            *,
            minecraft_version: str = "1.20.1",
            loader: str = "fabric",
            mappings: str = "yarn-1.20.1+build.1",
            limit: int = 6,
        ):
            # Preserve the exact legacy profile and its existing regression behavior.
            if (
                minecraft_version == "1.20.1"
                and mappings == "yarn-1.20.1+build.1"
            ):
                return original(
                    self,
                    query,
                    minecraft_version=minecraft_version,
                    loader=loader,
                    mappings=mappings,
                    limit=limit,
                )
            return _retrieve_live(
                retrieval_module,
                self,
                query,
                minecraft_version=minecraft_version,
                loader=loader,
                mappings=mappings,
                limit=limit,
            )

        retrieve._mmm_live_platform_rag = True
        cls.retrieve = retrieve

    def target_retrieve(retrieval: Any, query: str, *, adapter: Any, limit: int):
        with retrieval.OfficialCorpusIndex(documents=retrieval.BUILTIN_CORPUS) as index:
            return index.retrieve(
                query,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
                limit=limit,
            )

    target_retrieve._mmm_live_platform_rag = True
    # Replaces the old compatibility shim that queried a 1.20.1 source lane and then
    # rewrote the receipt target. No evidence is relabeled across versions anymore.
    platform_planning_module._target_retrieve = target_retrieve


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
        raise retrieval.SpecValidationError("The local official RAG lane supports Fabric only.")
    if not minecraft_version.strip() or not mappings.strip():
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
            f"No official corpus documents apply to Minecraft {minecraft_version}."
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

    family_hits = sum(
        family in eligible[hit.document_id].families for hit in hits
    )
    exact_hits = sum(
        minecraft_version in hit.minecraft_versions for hit in hits
    )
    applicable_hits = sum(
        minecraft_version in hit.minecraft_versions or "*" in hit.minecraft_versions
        for hit in hits
    )
    coverage = min(
        1.0,
        (applicable_hits / max(1, min(3, len(hits)))) * 0.55
        + (family_hits / max(1, min(3, len(hits)))) * 0.45,
    )
    # Version-neutral Fabric docs are valid conceptual evidence for a newly released
    # target; exact API correctness is established later by the official template,
    # JDT, Gradle and GameTest rather than by pretending an old Javadoc is exact.
    quality = "strong" if len(hits) >= min(3, limit) and family_hits >= 1 else "weak"
    correction_required = quality != "strong"
    corrections = (
        (
            f"{family} official Fabric documentation Minecraft {minecraft_version}",
            f"{family} {mappings} current symbol signature",
            f"{family} compile GameTest validation",
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
