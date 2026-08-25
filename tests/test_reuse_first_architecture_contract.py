from __future__ import annotations

from minecraft_mod_ai.reuse_discovery import discover_repositories_for_graph
from minecraft_mod_ai.reuse_planner import (
    CapabilityGraph,
    ReuseDecision,
    TargetImplementationPlan,
    decompose_capability_graph,
)


def test_detailed_request_catalog_is_primary_reuse_search_specification():
    design = {
        "_evidence_request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req-trade",
                    "capability": "economy.trade",
                    "statement": "NPC shops buy and sell goods using persistent currency balances",
                    "provides": ["economy.trade"],
                },
                {
                    "requirement_id": "req-cooking",
                    "capability": "cooking.recipe",
                    "statement": "Players cook ingredient recipes at a timed cooking station",
                    "provides": ["cooking.recipe"],
                },
            ]
        },
        "modules": ["unrelated_fallback_module"],
    }
    graph = decompose_capability_graph(
        "make a medieval life mod",
        design=design,
        module_kinds=("unrelated_fallback_module",),
    )
    assert graph.nodes == ("economy.trade", "cooking.recipe")
    payload = graph.to_dict()
    terms = {item["capability"]: item["terms"] for item in payload["search_terms"]}
    assert "NPC shops buy and sell goods using persistent currency balances" in terms["economy.trade"]
    assert "Players cook ingredient recipes at a timed cooking station" in terms["cooking.recipe"]
    assert all(source[1].startswith("evidence_request_catalog.") for source in graph.sources)


class _FakeDiscovery:
    def __init__(self):
        self.calls = []

    def search(self, provider, query, **kwargs):
        self.calls.append((provider, query, kwargs))
        if provider == "github":
            return {
                "candidates": [
                    {
                        "source_url": "https://github.com/example/shared-systems",
                        "title": "example/shared-systems",
                    },
                    {
                        "source_url": f"https://github.com/example/{query.split()[0].lower()}-feature",
                        "title": "feature",
                    },
                ]
            }
        return {"candidates": []}


def test_discovery_is_parallel_provider_aware_and_prefers_cross_capability_repositories(monkeypatch):
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    graph = CapabilityGraph(
        nodes=("economy.trade", "cooking.recipe"),
        edges=(),
        sources=(),
        search_terms=(
            ("economy.trade", ("merchant trading economy",)),
            ("cooking.recipe", ("cooking recipes station",)),
        ),
    )
    client = _FakeDiscovery()
    result = discover_repositories_for_graph(
        graph.nodes,
        client,
        capability_graph=graph.to_dict(),
    )
    assert result["economy.trade"][0] == "example/shared-systems"
    assert result["cooking.recipe"][0] == "example/shared-systems"
    providers = {provider for provider, _, _ in client.calls}
    assert providers == {"github", "modrinth"}


def test_target_plan_exposes_generation_scope_ledger():
    class Adapter:
        def public_dict(self):
            return {"minecraft_version": "1.21.1", "loader": "fabric"}

    decisions = (
        ReuseDecision("trade", "source_transplant", 0.9, 10, 4, source_id="a/b@1"),
        ReuseDecision("npc", "adapt", 0.8, 10, 4, source_id="a/b@1"),
        ReuseDecision("boss", "fresh", 1.0, 10, 4),
    )
    plan = TargetImplementationPlan(
        adapter=Adapter(),
        capabilities=decisions,
        platform_evidence=None,
        cross_component_integration_cost=0,
        platform_verification_cost=0,
        maintenance_risk=0,
        total_expected_cost=0,
        weighted_verified_reuse=0,
        fresh_work=0,
        adaptation_work=0,
        verification_work=0,
        uncertainty=0,
        reusable_registry_candidates=0,
    )
    ledger = {item["capability"]: item for item in plan.to_dict()["reuse_ledger"]}
    assert ledger["trade"]["fresh_generation_scope"] == "forbidden"
    assert ledger["npc"]["fresh_generation_scope"] == "residual_only"
    assert ledger["boss"]["fresh_generation_scope"] == "full"
