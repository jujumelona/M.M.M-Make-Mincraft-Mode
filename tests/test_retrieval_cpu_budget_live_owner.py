from __future__ import annotations

from dataclasses import dataclass

from minecraft_mod_ai import production_tools
from minecraft_mod_ai import retrieval_cpu_budget_contract as policy
from minecraft_mod_ai import small_model_hybrid_search_contract as hybrid


@dataclass(frozen=True)
class _Receipt:
    status: str = "FOUND"
    result_count: int = 0


@dataclass(frozen=True)
class _SearchResult:
    hits: tuple = ()
    receipt: _Receipt = _Receipt()


class _FakeIndex:
    search_calls: list[tuple[object, bool, bool]] = []
    build_calls: list[tuple[object, bool]] = []

    def __init__(self, _target) -> None:
        pass

    def search_with_receipt(
        self,
        _query,
        *,
        limit,
        router,
        semantic,
        rerank,
        required_metadata,
    ):
        del limit, required_metadata
        self.search_calls.append((router, bool(semantic), bool(rerank)))
        return _SearchResult()

    def build(self, _roots, *, metadata, router, semantic):
        del metadata
        self.build_calls.append((router, bool(semantic)))
        return {"status": "OK"}


def test_hybrid_dense_work_is_source_gated(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)

    assert hybrid._modes("semantic", False, False) == (
        (False, False, "lexical"),
    )
    assert hybrid._modes("dependency", False, False) == (
        (False, False, "lexical+relations"),
    )
    assert hybrid.adapt_query_vector(None, "q", ("x",)) == []

    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    monkeypatch.setattr(
        hybrid,
        "_adapt_query_vector_dense",
        lambda _router, _query, _texts, *, alpha: [alpha],
    )
    assert hybrid.adapt_query_vector(None, "q", ("x",), alpha=0.7) == [0.7]
    assert any(semantic or rerank for semantic, rerank, _label in hybrid._modes(
        "semantic", False, False
    ))


def test_production_tool_boundary_cannot_enable_dense_without_opt_in(
    monkeypatch,
    tmp_path,
) -> None:
    _FakeIndex.search_calls.clear()
    _FakeIndex.build_calls.clear()
    monkeypatch.setattr(production_tools, "ProjectRAGIndex", _FakeIndex)
    router = object()
    monkeypatch.setattr(
        production_tools,
        "ModelRouter",
        lambda *, profile: router,
    )

    service = production_tools.ProductionToolService(
        workspace_root=tmp_path,
        profile="test",
    )
    index_path = tmp_path / "rag" / "project-index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}", encoding="utf-8")

    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    service.search_code_rag("repair", semantic=True, rerank=True)
    service.index_project_rag(
        ("rag/project-index.json",),
        index_path="rag/rebuilt-index.json",
        metadata={},
        semantic=True,
    )
    assert _FakeIndex.search_calls[-1] == (None, False, False)
    assert _FakeIndex.build_calls[-1] == (None, False)

    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    service.search_code_rag("repair", semantic=True, rerank=True)
    service.index_project_rag(
        ("rag/project-index.json",),
        index_path="rag/rebuilt-index.json",
        metadata={},
        semantic=True,
    )
    assert _FakeIndex.search_calls[-1] == (router, True, True)
    assert _FakeIndex.build_calls[-1] == (router, True)


def test_cpu_budget_policy_has_no_runtime_installers() -> None:
    assert not hasattr(policy, "_install_live_hybrid_budget")
    assert not hasattr(policy, "_install_production_tool_budget")
    assert not hasattr(policy, "install")
