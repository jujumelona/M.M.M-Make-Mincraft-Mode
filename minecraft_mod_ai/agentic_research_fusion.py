from __future__ import annotations

"""Agentic evidence fusion for the target-aware Minecraft research lane.

The lower-level retriever already performs lexical/semantic/graph rank fusion.
This layer operates one level above it: independent research queries fan out in
parallel, weak receipts trigger bounded corrective retrieval, results are fused
per research domain, and dependency evidence is propagated as compact graph
context. The output remains evidence-only; it never authorizes code or assets.
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping


def _env_workers(name: str = "MMM_RESEARCH_WORKERS", default: int = 8) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(32, value))


def _receipt_dict(receipt: Any) -> dict[str, Any]:
    payload = receipt.to_dict()
    if not isinstance(payload, dict):
        raise TypeError("Retrieval receipt to_dict() must return an object.")
    return payload


def _receipt_quality(receipt: Any, payload: Mapping[str, Any]) -> str:
    value = payload.get("quality", getattr(receipt, "quality", "unknown"))
    return str(value or "unknown")


def _receipt_coverage(receipt: Any, payload: Mapping[str, Any]) -> float:
    value = payload.get("coverage", getattr(receipt, "coverage", 0.0))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _receipt_hits(receipt: Any, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("hits")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    hits = getattr(receipt, "hits", ())
    result: list[dict[str, Any]] = []
    for item in hits:
        if hasattr(item, "to_dict"):
            value = item.to_dict()
            if isinstance(value, Mapping):
                result.append(dict(value))
    return result


def _hit_key(hit: Mapping[str, Any]) -> str:
    for field in ("content_sha256", "document_id", "evidence_id", "url"):
        value = str(hit.get(field, "")).strip()
        if value:
            return f"{field}:{value}"
    return ""


def _fuse_domain(
    query_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    limit: int = 12,
) -> dict[str, Any]:
    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    support: dict[str, set[str]] = {}

    def add_receipt(
        receipt_payload: Mapping[str, Any],
        *,
        query_sha256: str,
        weight: float,
    ) -> None:
        raw_hits = receipt_payload.get("hits")
        if not isinstance(raw_hits, list):
            return
        for rank, raw_hit in enumerate(raw_hits, start=1):
            if not isinstance(raw_hit, Mapping):
                continue
            key = _hit_key(raw_hit)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            support.setdefault(key, set()).add(query_sha256)
            if key not in metadata:
                metadata[key] = {
                    "document_id": str(raw_hit.get("document_id", "")),
                    "evidence_id": str(raw_hit.get("evidence_id", "")),
                    "content_sha256": str(raw_hit.get("content_sha256", "")),
                    "url": str(raw_hit.get("url", "")),
                }

    qualities: list[str] = []
    coverages: list[float] = []
    primary_hit_count = 0
    correction_hit_count = 0
    correction_used = False

    for item in query_results:
        query_sha256 = str(item.get("query_sha256", ""))
        primary = item.get("primary")
        if isinstance(primary, Mapping):
            add_receipt(primary, query_sha256=query_sha256, weight=1.0)
            primary_hits = primary.get("hits")
            if isinstance(primary_hits, list):
                primary_hit_count += len(primary_hits)
            qualities.append(str(primary.get("quality", "unknown")))
            try:
                coverages.append(float(primary.get("coverage", 0.0)))
            except (TypeError, ValueError):
                coverages.append(0.0)

        corrections = item.get("corrections")
        if isinstance(corrections, list):
            correction_used = correction_used or bool(corrections)
            for correction in corrections:
                if not isinstance(correction, Mapping):
                    continue
                add_receipt(correction, query_sha256=query_sha256, weight=0.72)
                correction_hits = correction.get("hits")
                if isinstance(correction_hits, list):
                    correction_hit_count += len(correction_hits)
                qualities.append(str(correction.get("quality", "unknown")))
                try:
                    coverages.append(float(correction.get("coverage", 0.0)))
                except (TypeError, ValueError):
                    coverages.append(0.0)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    fused = [
        {
            **metadata[key],
            "rrf_score": round(scores[key], 10),
            "query_support": sorted(support.get(key, set())),
        }
        for key in ordered
    ]

    total_hits = primary_hit_count + correction_hit_count
    strong_receipts = sum(value == "strong" for value in qualities)
    mean_coverage = (
        sum(max(0.0, min(1.0, value)) for value in coverages) / len(coverages)
        if coverages
        else 0.0
    )
    if total_hits == 0:
        decision = "reflect"
    elif correction_used:
        decision = "corrected_accept"
    else:
        decision = "accept"

    return {
        "schema_version": "mmm/agentic-domain-fusion-v1",
        "method": "weighted_reciprocal_rank_fusion",
        "rrf_k": rrf_k,
        "documents": fused,
        "critic": {
            "decision": decision,
            "receipt_count": len(qualities),
            "strong_receipts": strong_receipts,
            "primary_hit_count": primary_hit_count,
            "correction_hit_count": correction_hit_count,
            "mean_coverage": round(mean_coverage, 6),
            "needs_reflection": decision == "reflect",
        },
    }


def retrieve_target_agentic_evidence(
    research_brief: dict[str, Any],
    *,
    central_module: Any,
    retrieve: Callable[..., Any],
    minecraft_version: str,
    loader: str,
    mappings: str,
    include_target: bool = True,
) -> dict[str, Any]:
    """Run target-aware adaptive RAG with deterministic parallel fan-out.

    Research domains act as specialist lanes. Primary queries are independent and
    execute concurrently. Only receipts that request correction spawn second-hop
    retrieval. Per-domain RRF then compacts overlapping evidence, and dependency
    edges expose upstream fused document IDs without duplicating excerpts.
    """

    raw_domains = research_brief.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise central_module.SpecValidationError("Central research brief has no domains.")
    domains = [central_module._research_domain(raw) for raw in raw_domains]

    jobs: list[tuple[int, int, str]] = []
    for domain_index, domain in enumerate(domains):
        if "official_docs" not in domain.providers:
            continue
        for query_index, query in enumerate(domain.queries):
            jobs.append((domain_index, query_index, query))

    workers = min(_env_workers(), max(1, len(jobs)))
    primary_results: dict[tuple[int, int], Any] = {}
    correction_results: dict[tuple[int, int], list[Any | None]] = {}
    correction_job_count = 0

    def fetch_primary(job: tuple[int, int, str]) -> tuple[int, int, Any]:
        domain_index, query_index, query = job
        receipt = retrieve(
            query,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            limit=8,
        )
        return domain_index, query_index, receipt

    if jobs:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mmm_agentic_rag",
        ) as pool:
            primary_futures = [pool.submit(fetch_primary, job) for job in jobs]
            for future in as_completed(primary_futures):
                domain_index, query_index, primary = future.result()
                primary_results[(domain_index, query_index)] = primary

            correction_futures: dict[Future[Any], tuple[int, int, int]] = {}
            for domain_index, query_index, _query in jobs:
                primary = primary_results[(domain_index, query_index)]
                queries = tuple(getattr(primary, "correction_queries", ()) or ())
                correction_results[(domain_index, query_index)] = [None] * len(queries)
                for correction_index, correction_query in enumerate(queries):
                    future = pool.submit(
                        retrieve,
                        correction_query,
                        minecraft_version=minecraft_version,
                        loader=loader,
                        mappings=mappings,
                        limit=4,
                    )
                    correction_futures[future] = (
                        domain_index,
                        query_index,
                        correction_index,
                    )
                    correction_job_count += 1

            for future in as_completed(correction_futures):
                domain_index, query_index, correction_index = correction_futures[future]
                correction_results[(domain_index, query_index)][
                    correction_index
                ] = future.result()

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    fused_by_domain: dict[str, dict[str, Any]] = {}

    for domain_index, domain in enumerate(domains):
        if "official_docs" not in domain.providers:
            results.append(
                {
                    "domain_id": domain.domain_id,
                    "strategy": "routed_to_other_providers",
                    "retrieval_decision": "skip_official_lane",
                    "queries": [],
                    "research_methods": [
                        "specialist_domain_routing",
                        "provider_gated_adaptive_retrieval",
                    ],
                }
            )
            continue

        query_results: list[dict[str, Any]] = []
        has_hits = False
        for query_index, query in enumerate(domain.queries):
            primary = primary_results[(domain_index, query_index)]
            primary_payload = _receipt_dict(primary)
            corrections: list[dict[str, Any]] = []
            for item in correction_results.get((domain_index, query_index), []):
                if item is None:
                    continue
                corrections.append(_receipt_dict(item))

            primary_hits = _receipt_hits(primary, primary_payload)
            correction_hits = [
                hit
                for correction in corrections
                for hit in (
                    correction.get("hits")
                    if isinstance(correction.get("hits"), list)
                    else []
                )
                if isinstance(hit, Mapping)
            ]
            has_hits = has_hits or bool(primary_hits) or bool(correction_hits)
            correction_required = bool(
                getattr(primary, "correction_required", False)
                or primary_payload.get("correction_required")
                or getattr(primary, "correction_queries", ())
            )
            query_results.append(
                {
                    "query_sha256": central_module._sha256(query),
                    "strategy": (
                        "single"
                        if not correction_required
                        else "corrective_multi_hop"
                    ),
                    "retrieval_critic": {
                        "quality": _receipt_quality(primary, primary_payload),
                        "coverage": round(
                            _receipt_coverage(primary, primary_payload), 6
                        ),
                        "correction_required": correction_required,
                        "primary_hit_count": len(primary_hits),
                        "correction_hit_count": len(correction_hits),
                    },
                    "primary": primary_payload,
                    "corrections": corrections,
                }
            )

        fusion = _fuse_domain(query_results)
        fused_by_domain[domain.domain_id] = fusion
        if not has_hits:
            unresolved.append(domain.domain_id)
        results.append(
            {
                "domain_id": domain.domain_id,
                "strategy": "agentic_adaptive_parallel",
                "retrieval_decision": "retrieve",
                "research_methods": [
                    "specialist_domain_fanout",
                    "provider_gated_adaptive_retrieval",
                    "corrective_retrieval",
                    "weighted_reciprocal_rank_fusion",
                    "evidence_self_critique",
                    "dependency_graph_propagation",
                ],
                "queries": query_results,
                "fusion": fusion,
            }
        )

    result_by_id = {
        str(item.get("domain_id", "")): item
        for item in results
        if isinstance(item, Mapping)
    }
    for domain in domains:
        if not domain.depends_on:
            continue
        dependency_context: list[dict[str, Any]] = []
        for dependency_id in domain.depends_on:
            dependency_fusion = fused_by_domain.get(dependency_id)
            if not dependency_fusion:
                continue
            documents = dependency_fusion.get("documents")
            document_ids = []
            if isinstance(documents, list):
                document_ids = [
                    str(item.get("document_id", ""))
                    for item in documents
                    if isinstance(item, Mapping) and item.get("document_id")
                ]
            dependency_context.append(
                {
                    "domain_id": dependency_id,
                    "document_ids": document_ids,
                }
            )
        target_result = result_by_id.get(domain.domain_id)
        if target_result is not None:
            target_result["dependency_evidence"] = dependency_context

    payload: dict[str, Any] = {
        "schema_version": "mmm/central-evidence-graph-v1",
        "brief_sha256": research_brief.get("brief_sha256", ""),
        "domains": results,
        "unresolved_official_domains": unresolved,
        "authorization": "none",
        "retrieval_is_authority": False,
        "agentic_research": {
            "schema_version": "mmm/agentic-research-runtime-v1",
            "parallel": len(jobs) > 1 and workers > 1,
            "workers": workers if jobs else 0,
            "primary_jobs": len(jobs),
            "correction_jobs": correction_job_count,
            "deterministic_merge_order": True,
            "methods": [
                "adaptive_retrieval",
                "corrective_retrieval",
                "hybrid_rank_fusion",
                "graph_context_propagation",
                "specialist_agent_fanout",
                "evidence_critique",
            ],
        },
    }
    if include_target:
        payload["target"] = {
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
        }
    payload["evidence_sha256"] = central_module._sha256(
        central_module.canonical_json(payload)
    )
    return payload


__all__ = [
    "retrieve_target_agentic_evidence",
]
