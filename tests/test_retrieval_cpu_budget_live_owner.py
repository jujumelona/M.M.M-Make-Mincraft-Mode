from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import retrieval_cpu_budget_contract as policy


def _dense_modes(_route: str, _semantic: bool, _rerank: bool):
    return (
        (False, False, "lexical"),
        (False, True, "lexical+rerank"),
        (True, True, "semantic+rerank"),
    )


def test_live_hybrid_guard_rebinds_after_late_owner_replacement(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    hybrid = SimpleNamespace(
        _modes=_dense_modes,
        adapt_query_vector=lambda _router, _query, _texts: [1.0],
    )

    policy._install_live_hybrid_budget(hybrid)
    assert hybrid._modes("semantic", False, False) == ((False, False, "lexical"),)
    assert hybrid.adapt_query_vector(None, "q", ("x",)) == []

    # Simulate a later runtime composer replacing both executable owners while an old
    # module-level installation marker would still be present.
    hybrid._modes = _dense_modes
    hybrid.adapt_query_vector = lambda _router, _query, _texts: [2.0]
    hybrid._mmm_cpu_dense_hybrid_guard_v1 = True

    policy._install_live_hybrid_budget(hybrid)
    assert hybrid._modes("dependency", False, False) == (
        (False, False, "lexical+relations"),
    )
    assert hybrid.adapt_query_vector(None, "q", ("x",)) == []


def test_production_tool_boundary_cannot_enable_dense_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)

    class Service:
        def __init__(self) -> None:
            self.search_calls: list[tuple[bool, bool]] = []
            self.index_calls: list[bool] = []

        def search_code_rag(
            self,
            query: str,
            *,
            index_path: str = "rag/project-index.json",
            limit: int = 8,
            semantic: bool = False,
            rerank: bool = False,
            required_metadata=None,
        ):
            del query, index_path, limit, required_metadata
            self.search_calls.append((semantic, rerank))
            return {"ok": True}

        def index_project_rag(
            self,
            roots,
            *,
            index_path: str = "rag/project-index.json",
            metadata,
            semantic: bool = False,
        ):
            del roots, index_path, metadata
            self.index_calls.append(semantic)
            return {"ok": True}

    module = SimpleNamespace(ProductionToolService=Service)
    policy._install_production_tool_budget(module)
    service = Service()

    service.search_code_rag("repair", semantic=True, rerank=True)
    service.index_project_rag(("src",), metadata={}, semantic=True)
    assert service.search_calls == [(False, False)]
    assert service.index_calls == [False]

    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    service.search_code_rag("repair", semantic=True, rerank=True)
    service.index_project_rag(("src",), metadata={}, semantic=True)
    assert service.search_calls[-1] == (True, True)
    assert service.index_calls[-1] is True
