from __future__ import annotations

from minecraft_mod_ai.small_model_hybrid_search_contract import _modes


def test_exact_and_dependency_routes_start_without_embedding_or_reranker() -> None:
    for route in ("exact_version", "exact_symbol", "dependency", "global"):
        semantic, rerank, _mode = _modes(route, True, True)[0]
        assert semantic is False
        assert rerank is False


def test_dependency_route_defaults_to_cheap_graph_only(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    assert _modes("dependency", False, False) == (
        (False, False, "lexical+relations"),
    )


def test_dependency_dense_escalation_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    modes = _modes("dependency", False, False)
    assert modes[0] == (False, False, "lexical+relations")
    assert modes[1] == (False, True, "lexical+rerank+relations")
    assert modes[2] == (True, True, "semantic+rerank+relations")


def test_free_semantic_queries_default_to_lexical_only(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    assert _modes("semantic", False, False) == ((False, False, "lexical"),)


def test_free_semantic_dense_escalation_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    modes = _modes("semantic", False, False)
    assert modes[0] == (False, False, "lexical")
    assert modes[1] == (False, True, "lexical+rerank")
    assert modes[2] == (True, True, "semantic+rerank")
