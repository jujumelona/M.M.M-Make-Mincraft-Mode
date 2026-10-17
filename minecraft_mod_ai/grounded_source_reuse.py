from __future__ import annotations
"""CurseForge catalogue research handoff; no external source repositories."""
import json
from collections.abc import Mapping
from typing import Any
from .curseforge_catalog import curseforge_api_key
from .platform_catalog import adapter_for_target
from .reuse_discovery import discover_repositories_for_graph

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
    for name in capabilities:
        row = receipts_by_cap.get(name) or single_legacy_receipt or {}
        discovery_status = str(row.get("status") or "query_unresolved")
        matched = list(row.get("ranked_catalogue_hits") or ())
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
        decisions.append({
            "capability": name, "mode": "fresh", "search_status": status,
            "discovery_status": discovery_status,
            "failure_code": failure_code, "source_id": "",
            "component_refs": [], "reference_source_count": 0,
            "catalogue_evidence": matched[:5],
            "catalogue_queries": list(row.get("queries") or ()),
            "rationale": (
                "CurseForge catalogue metadata is only a research hint; "
                "implement behavior against the selected Minecraft target."
            ),
        })
        print("SOURCE_REUSE_CAPABILITY: " + json.dumps({
            "capability": name, "mode": "fresh", "inspected": 0,
            "proof_attempts": 0, "status": discovery_status,
            "evidence_status": status, "catalogue_match_count": len(matched),
            "queries": list(row.get("queries") or ()),
        }, ensure_ascii=False), flush=True)
    return {
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "bound_target": {"minecraft_version": version, "loader": loader},
        "source_plan_sha256": sha, "capability_graph": dict(graph),
        "capabilities": decisions, "grounded_repository_count": 0,
        "host_discovered_repositories": {
            name: list(found.get(name, ())) for name in capabilities
        },
        "planning_rag_candidate_repository_count": 0,
        "explicit_reference_repository_count": 0,
        "discovery_receipts": receipts, "inspection_receipts": [],
        "proof_receipts": [], "reference_sources": [],
        "selection_policy": "Targeted CurseForge metadata only; independent fresh Java code.",
    }


__all__ = ["build_repository_reuse_plan"]
