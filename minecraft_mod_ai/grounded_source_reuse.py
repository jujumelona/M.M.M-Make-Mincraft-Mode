from __future__ import annotations
"""CurseForge catalogue research handoff; no external source repositories."""
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any
from .curseforge_catalog import curseforge_api_key
from .platform_catalog import adapter_for_target
from .reuse_discovery import discover_repositories_for_graph
from .curseforge_semantic_source_review import assess_ranked_sources
from .source_inspection_recovery import usable_source_links, unusable_source_links

def build_repository_reuse_plan(design: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Produce fresh-authoring decisions from catalogue observations only."""
    plan = design.get("_pre_retrieval_plan")
    if not isinstance(plan, Mapping):
        raise TypeError("Missing frozen pre-retrieval plan")
    sha = str(plan.get("plan_sha256") or "")
    graph = plan.get("capability_graph")
    if not sha or not isinstance(graph, Mapping):
        raise ValueError("Incomplete frozen plan")
    selection = design.get("_platform_selection")
    target = selection.get("target") if isinstance(selection, Mapping) else None
    if not isinstance(target, Mapping):
        raise TypeError("Missing selected target")
    version = str(target.get("minecraft_version") or "").strip()
    loader = str(target.get("loader") or "").strip().casefold()
    if not version or not loader:
        raise ValueError("Incomplete selected target")
    adapter_for_target(version, loader).validate()
    capabilities = tuple(dict.fromkeys(str(node).strip() for node in graph.get("nodes", ()) if str(node).strip()))
    if not capabilities:
        raise ValueError("Missing frozen capability nodes")
    receipts = []
    found = discover_repositories_for_graph(capabilities, None, capability_graph=graph, minecraft_version=version, loader=loader, diagnostics=receipts, providers=("curseforge",), query_cache={})
    print("SOURCE_REUSE_DISCOVERY_ROUTE: " + json.dumps({"provider_order":["curseforge"], "curseforge_key_present":bool(curseforge_api_key()), "minecraft_version":version, "loader":loader}), flush=True)
    # Source evidence is decided independently for every feature. A positive
    # search for one capability must never cover an unrelated requirement.
    receipts_by_cap = {
        str(row.get("capability") or ""): row for row in receipts
    }
    single_legacy_receipt = receipts[0] if len(capabilities) == len(receipts) == 1 else None
    decisions = []
    inspection_receipts = []
    reference_sources = []
    shared_source_cache, unusable_source_urls = {}, set()
    remaining_inspections = max(8, len(capabilities))  # One fair pass per feature; no speculative double scan.
    for index, name in enumerate(capabilities):
        row = receipts_by_cap.get(name) or single_legacy_receipt or {}
        discovery_status = str(row.get("status") or "query_unresolved")
        matched = list(row.get("ranked_catalogue_hits") or ())
        available_links = usable_source_links(matched, unusable_source_urls)
        # Reserve capacity for each outstanding capability before considering
        # a second donor. This prevents early catalogue matches from starving
        # later, unrelated requirements.
        outstanding = len(capabilities) - index - 1
        allowance = min(2, max(0, remaining_inspections - outstanding))
        inspections = (
            assess_ranked_sources(name, available_links, router=kwargs.get("router"),
                                  cache=shared_source_cache,
                                  max_candidates=allowance)
            if allowance else []
        )
        remaining_inspections -= len(inspections)
        inspection_receipts.extend(inspections)
        unusable_source_urls.update(unusable_source_links(inspections))
        # Examined candidate != transplant. Only compile-proof may set source_transplant.
        for candidate in inspections:
            semantic = candidate.get("semantic") or {}
            if (candidate.get("inspection_status") == "inspected"
                    and semantic.get("status") == "semantic_reviewed"
                    and semantic.get("intent") in {"reference", "adapt", "reuse"}):
                reference_sources.append({
                    "capability": name, "source_id": candidate.get("source_id"),
                    "source_url": candidate.get("source_url"),
                    "pinned_sha": candidate.get("pinned_sha"),
                    "license_spdx": candidate.get("license_spdx"),
                    "selected_file": semantic.get("selected_file"),
                    "selected_symbol": semantic.get("selected_symbol"),
                    "file_sha256": semantic.get("file_sha256"),
                    "semantic_score": semantic.get("semantic_score"),
                    "semantic_intent": semantic.get("intent"),
                    "semantic_scores": semantic.get("scores") or {},
                    "rationale": semantic.get("rationale"),
                    "residual_work": semantic.get("residual_work"),
                    "source_excerpt": candidate.get("licensed_reference_excerpt", ""),
                    "proof_status": "NOT_RUN", "reference_only": True,
                })
        reference_count = sum(ref["capability"] == name for ref in reference_sources)
        if discovery_status in {"provider_error", "not_configured"}:
            status, failure_code = ("provider_error",
                "CURSEFORGE_NOT_CONFIGURED" if discovery_status == "not_configured"
                else "CURSEFORGE_PROVIDER_ERROR")
        elif discovery_status == "catalog_only" and matched:
            status, failure_code = "unresolved", ""
        elif discovery_status == "catalog_only" and row.get("catalog_result_count", 0):
            status, failure_code = "unresolved", ""
        elif discovery_status == "search_empty":
            status, failure_code = "search_empty", ""
        else:
            status, failure_code = "unresolved", ""
        if reference_count and status == "unresolved":
            status = "reference_available"
        decisions.append({
            "capability": name, "mode": "fresh", "search_status": status,
            "discovery_status": discovery_status,
            "failure_code": failure_code, "source_id": "",
            "component_refs": [], "reference_source_count": reference_count,
            "catalogue_evidence": matched,  # Already bounded per distinct topic.
            "semantic_assessments": inspections,
            "catalogue_queries": list(row.get("queries") or ()),
            "rationale": (
                "Catalogue metadata ranks candidates. Official linked source is "
                "inspected and semantically reviewed when available. "
                "Unproved donors remain reference-only; complete fresh implementation."
            ),
        })
        print("SOURCE_REUSE_CAPABILITY: " + json.dumps({
            "stage": "planning", "capability": name, "mode": "fresh", "inspected": sum(
                item.get("inspection_status") == "inspected" for item in inspections),
            "source_links": len(available_links),
            "semantic_reviews": sum((item.get("semantic") or {}).get("status")
                                    == "semantic_reviewed" for item in inspections),
            "proof_attempts": 0, "status": discovery_status,
            "evidence_status": status, "catalogue_match_count": len(matched),
            "reference_source_count": reference_count,
            "inspection_outcomes": [
                {"source_id": item.get("source_id"),
                 "status": item.get("inspection_status"),
                 "semantic_status": (item.get("semantic") or {}).get("status"),
                 "semantic_intent": (item.get("semantic") or {}).get("intent"),
                 "error": item.get("error") or (item.get("semantic") or {}).get("error", "")}
                for item in inspections
            ],
            "queries": list(row.get("queries") or ()),
        }, ensure_ascii=False), flush=True)
    # Explicit local CurseForge dependency selections remain UNVERIFIED here.
    # Actual authenticated file lookup, license acknowledgement, byte hash,
    # Fabric metadata and Gradle wiring happen at project materialization.
    requested_dependencies = design.get("_requested_curseforge_dependencies", ())
    if not isinstance(requested_dependencies, (list, tuple)):
        raise ValueError("SOURCE_REUSE_DEPENDENCY_SELECTION_INVALID")
    requested_dependencies = [
        {k: item[k] for k in (
            "project_id", "file_id", "jar_path", "license_policy_confirmed")
         if k in item}
        for item in requested_dependencies if isinstance(item, Mapping)
    ][:8]
    performance = {
        "source_downloads": sum(int(x.get("source_fetches") or 0) for x in inspection_receipts),
        "source_cache_hits": sum(int(x.get("source_cache_hits") or 0) for x in inspection_receipts),
        "model_judgement_calls": sum(int(x.get("model_judgement_calls") or 0) for x in inspection_receipts),
        "semantic_cache_hits": sum(int(x.get("semantic_cache_hits") or 0) for x in inspection_receipts),
        "inspection_wall_ms": round(sum(float(x.get("elapsed_ms") or 0) for x in inspection_receipts), 1),
        "inspected_candidate_count": len(inspection_receipts),
        "successful_source_inspections": sum(
            item.get("inspection_status") == "inspected" for item in inspection_receipts),
        "inspection_status_counts": dict(Counter(
            str(item.get("inspection_status") or "unknown") for item in inspection_receipts)),
        "capability_count": len(capabilities),
    }
    print("SOURCE_REUSE_COSTS: " + json.dumps(performance, ensure_ascii=False), flush=True)
    return {
        "source_reuse_costs": performance,
        "requested_curseforge_dependencies": requested_dependencies,
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "bound_target": {"minecraft_version": version, "loader": loader},
        "source_plan_sha256": sha, "capability_graph": dict(graph),
        "capabilities": decisions, "grounded_repository_count": 0,
        "host_discovered_repositories": {
            name: list(found.get(name, ())) for name in capabilities
        },
        "planning_rag_candidate_repository_count": 0,
        "explicit_reference_repository_count": 0,
        "discovery_receipts": receipts, "inspection_receipts": inspection_receipts,
        "proof_receipts": [], "reference_sources": reference_sources,
        "selection_policy": "CurseForge official source inspection + model semantic reference review; "
                            "reuse needs isolated compile proof and installation.",
    }


__all__ = ["build_repository_reuse_plan"]
