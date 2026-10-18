from __future__ import annotations



from minecraft_mod_ai.canonical_capability_ontology import (
    canonical_domain_map,
    resolve_capabilities_from_phrase,
    resolve_capabilities_from_phrase_structured,
)
from minecraft_mod_ai.capability_semantic_inference import (
    enrich_resolution_with_semantic_inference,
)


def test_unified_ontology_contains_reviewed_domain_terms() -> None:
    ontology_map = canonical_domain_map()
    assert "trade" in ontology_map
    assert "boss" in ontology_map
    assert "nuclear" in ontology_map
    assert "medieval" in ontology_map


def test_ontology_can_expand_theme_before_request_catalog_approval() -> None:
    caps = resolve_capabilities_from_phrase("medieval mod")
    assert "trade.shop_registry" in caps
    assert "economy.currency" in caps
    assert "quest.state" in caps
    assert "worldgen.structure" in caps


def test_unresolved_concept_requires_explicit_semantic_evidence() -> None:
    res = resolve_capabilities_from_phrase_structured("medieval banking loan system")
    assert res.unresolved_spans
    unevidenced = enrich_resolution_with_semantic_inference(res)
    assert not any(node.capability_id.startswith("provisional:") for node in unevidenced.nodes)

    class SemanticRouter:
        def generate_text(self, role, messages, **kwargs):
            assert role == "planner"
            return {"capabilities": [{
                "capability_id": "bank.loan",
                "source_span": res.unresolved_spans[0],
                "category": "finance",
                "description": "Bank lending system",
                "dependencies": ["persistence.state_store"],
            }]}

    enriched = enrich_resolution_with_semantic_inference(
        res, router=SemanticRouter(),
    )
    assert any(node.capability_id == "bank.loan" for node in enriched.nodes)
    assert any("persistence.state_store" in edge[1] for edge in enriched.edges)


