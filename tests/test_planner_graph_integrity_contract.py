from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.reuse_planner import decompose_capability_graph


def _catalog() -> dict[str, object]:
    return {
        "requirements": [
            {
                "requirement_id": "req-c",
                "capability": "feature.c",
                "provides": ["capability:feature.c"],
                "statement": "feature c",
                "depends_on": [],
            },
            {
                "requirement_id": "req-b",
                "capability": "feature.b",
                "provides": ["capability:feature.b"],
                "statement": "feature b",
                "depends_on": ["req-c"],
            },
            {
                "requirement_id": "req-a",
                "capability": "feature.a",
                "provides": ["capability:feature.a"],
                "statement": "feature a",
                "depends_on": ["req-b"],
            },
        ]
    }


def test_authored_requirement_dependencies_are_the_graph_authority() -> None:
    graph = decompose_capability_graph(
        "prompt text cannot invent graph edges",
        design={"_evidence_request_catalog": _catalog()},
    )
    assert graph.nodes == ("feature.c", "feature.b", "feature.a")
    assert graph.edges == (
        ("feature.b", "feature.c"),
        ("feature.a", "feature.b"),
    )


def test_graph_sources_and_search_terms_cover_every_node() -> None:
    graph = decompose_capability_graph(
        "ignored after request approval",
        design={"_evidence_request_catalog": _catalog()},
    )
    assert {capability for capability, _source in graph.sources} == set(graph.nodes)
    assert {capability for capability, _terms in graph.search_terms} == set(graph.nodes)
    assert all(terms for _capability, terms in graph.search_terms)


def test_retired_runtime_graph_monkeypatch_owner_is_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert not (package / "planner_graph_integrity_contract.py").exists()
