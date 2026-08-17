from __future__ import annotations

from minecraft_mod_ai.small_model_hybrid_search_contract import _modes


def test_exact_and_dependency_routes_start_without_embedding_or_reranker() -> None:
    for route in ("exact_version", "exact_symbol", "dependency", "global"):
        semantic, rerank, _mode = _modes(route, True, True)[0]
        assert semantic is False
        assert rerank is False


def test_dependency_route_escalates_cost_only_after_cheap_graph_pass() -> None:
    modes = _modes("dependency", False, False)
    assert modes[0] == (False, False, "lexical+relations")
    assert modes[1] == (False, True, "lexical+rerank+relations")
    assert modes[2] == (True, True, "semantic+rerank+relations")


def test_free_semantic_queries_keep_semantic_first() -> None:
    assert _modes("semantic", False, False)[0] == (
        True,
        True,
        "semantic+rerank",
    )
