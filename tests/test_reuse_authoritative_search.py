from __future__ import annotations

from minecraft_mod_ai.requirement_catalog import build_requirement_catalog
from minecraft_mod_ai.reuse_planner import decompose_capability_graph


class _ExplodingSemanticRouter:
    def generate_text(self, *args, **kwargs):
        raise AssertionError("raw prompt semantic inference ran after catalog approval")

    def generate_tool_decision(self, *args, **kwargs):
        raise AssertionError("raw prompt semantic inference ran after catalog approval")


def _catalog():
    return {
        "requirements": [
            {
                "requirement_id": "req_collect",
                "capability": "resource.gathering",
                "provides": ["capability:resource.gathering"],
                "statement": "Broad authored text that must not own donor search semantics.",
                "semantic_statement": "collect luminous shards from world nodes",
                "source_span": {"text": "collect luminous shards"},
                "mandatory": True,
                "depends_on": [],
            },
            {
                "requirement_id": "req_exchange",
                "capability": "economy.exchange",
                "provides": ["capability:economy.exchange"],
                "statement": "Another broad authored sentence.",
                "semantic_statement": "exchange gathered shards for progression value",
                "source_span": {"text": "exchange gathered shards"},
                "mandatory": True,
                "depends_on": ["req_collect"],
            },
        ]
    }


def test_approved_catalog_is_hard_authority_barrier_for_reuse_search():
    graph = decompose_capability_graph(
        "NEVER_SEARCH_THIS_THEME arbitrary unrelated prompt wording",
        design={"_evidence_request_catalog": _catalog()},
        semantic_router=_ExplodingSemanticRouter(),
    )

    assert graph.nodes == ("resource.gathering", "economy.exchange")
    assert graph.edges == (("economy.exchange", "resource.gathering"),)
    assert all(not node.startswith("provisional:semantic_") for node in graph.nodes)

    terms = dict(graph.search_terms)
    flattened = " ".join(term for values in terms.values() for term in values)
    assert "NEVER_SEARCH_THIS_THEME" not in flattened
    assert "collect luminous shards from world nodes" in flattened
    assert "exchange gathered shards for progression value" in flattened
    assert all(
        capability.replace(".", " ") in " ".join(values)
        for capability, values in terms.items()
    )


def test_authoritative_requirement_catalog_prefers_semantic_statement():
    catalog = build_requirement_catalog(
        "raw prompt is not the normalized search meaning",
        evidence_request_catalog=_catalog(),
    )
    by_id = {item.id: item for item in catalog.requirements}
    assert by_id["req_collect"].normalized_statement == "collect luminous shards from world nodes"
    assert by_id["req_exchange"].normalized_statement == "exchange gathered shards for progression value"
