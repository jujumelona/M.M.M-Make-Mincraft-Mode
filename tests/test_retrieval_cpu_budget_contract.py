from __future__ import annotations

from dataclasses import dataclass

import pytest

from minecraft_mod_ai import production_tools
from minecraft_mod_ai import retrieval_cpu_budget_contract as policy
from minecraft_mod_ai import small_model_hybrid_search_contract as hybrid
from minecraft_mod_ai.model_adapters import embedding as embedding_module
from minecraft_mod_ai.model_adapters import reranker as reranker_module
from minecraft_mod_ai.model_adapters.base import AdapterConfig, ModelConfigurationError


def _retrieval_config(*, role: str, adapter: str, model_id: str) -> AdapterConfig:
    return AdapterConfig(
        role=role,
        adapter=adapter,
        model_id=model_id,
        max_context=512,
        extra={"device": "cpu"},
    )


def test_dense_cpu_retrieval_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    assert policy._dense_opted_in() is False

    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    assert policy._dense_opted_in() is True


def test_hybrid_modes_are_source_owned_lexical_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)

    assert hybrid._modes("exact_symbol", True, True) == (
        (False, False, "lexical"),
    )
    assert hybrid._modes("dependency", True, True) == (
        (False, False, "lexical+relations"),
    )
    assert hybrid._modes("global", True, True) == (
        (False, False, "lexical+global-relations"),
    )


def test_hybrid_centroid_adaptation_does_not_touch_dense_backend_without_opt_in(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    monkeypatch.setattr(
        hybrid,
        "_adapt_query_vector_dense",
        lambda *args, **kwargs: pytest.fail("dense adaptation must not run"),
    )

    assert hybrid.adapt_query_vector(object(), "query", ["hit"]) == []


def test_production_search_forces_lexical_without_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    captured: dict[str, object] = {}

    @dataclass
    class Receipt:
        result_count: int = 0

    class Result:
        hits = ()
        receipt = Receipt()

    class FakeIndex:
        def __init__(self, target):
            captured["target"] = target

        def search_with_receipt(
            self,
            query,
            *,
            limit,
            router,
            semantic,
            rerank,
            required_metadata,
        ):
            captured.update(
                {
                    "query": query,
                    "limit": limit,
                    "router": router,
                    "semantic": semantic,
                    "rerank": rerank,
                    "required_metadata": required_metadata,
                }
            )
            return Result()

    monkeypatch.setattr(production_tools, "ProjectRAGIndex", FakeIndex)
    service = production_tools.ProductionToolService(workspace_root=tmp_path)
    result = service.search_code_rag(
        "find symbol",
        semantic=True,
        rerank=True,
    )

    assert result["hits"] == []
    assert captured["semantic"] is False
    assert captured["rerank"] is False
    assert captured["router"] is None


def test_embedding_loader_fails_closed_before_dependency_or_model_load(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    monkeypatch.setattr(
        embedding_module,
        "require_package",
        lambda *args, **kwargs: pytest.fail("embedding dependency check must not run"),
    )
    adapter = embedding_module.EmbeddingAdapter(
        _retrieval_config(
            role="retrieval_embedding",
            adapter="embedding",
            model_id="Qwen/Qwen3-Embedding-0.6B",
        )
    )

    with pytest.raises(ModelConfigurationError, match="MMM_RAG_ENABLE_CPU_DENSE=1"):
        adapter._load_backend()


def test_reranker_loader_fails_closed_before_dependency_or_model_load(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    monkeypatch.setattr(
        reranker_module,
        "require_package",
        lambda *args, **kwargs: pytest.fail("reranker dependency check must not run"),
    )
    adapter = reranker_module.RerankerAdapter(
        _retrieval_config(
            role="retrieval_reranker",
            adapter="reranker",
            model_id="Qwen/Qwen3-Reranker-0.6B",
        )
    )

    with pytest.raises(ModelConfigurationError, match="MMM_RAG_ENABLE_CPU_DENSE=1"):
        adapter._load_backend()
