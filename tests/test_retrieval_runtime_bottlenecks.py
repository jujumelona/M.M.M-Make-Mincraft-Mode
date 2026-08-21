from __future__ import annotations

from functools import wraps
from types import SimpleNamespace

from minecraft_mod_ai.retrieval_adaptation import adapt_query_vector
from minecraft_mod_ai.small_model_retrieval_efficiency_contract import (
    _install_explicit_semantic_index_policy,
    _install_pre_design_rag_cascade,
)


def test_adaptation_embeds_records_not_string_characters() -> None:
    calls: list[list[str]] = []

    class Router:
        def embed(self, texts):
            values = list(texts)
            calls.append(values)
            return [
                [float(index + 1), 1.0, 0.5]
                for index, _text in enumerate(values)
            ]

    vector = adapt_query_vector(
        Router(),
        "repair source",
        ["first hit", "second hit", "third hit"],
    )

    assert vector
    assert calls == [
        ["repair source"],
        ["first hit", "second hit", "third hit"],
    ]


def test_implicit_repair_semantic_index_is_not_forced() -> None:
    calls: list[bool] = []

    def base(
        self,
        roots,
        *,
        index_path="rag/project-index.json",
        metadata,
        semantic=False,
    ):
        del self, roots, index_path, metadata
        calls.append(bool(semantic))
        return {"semantic": bool(semantic)}

    @wraps(base)
    def forced(
        self,
        roots,
        *,
        index_path="rag/project-index.json",
        metadata,
        semantic=False,
    ):
        repair_like = bool(metadata.get("source_commit")) and metadata.get("license") == "project-local"
        return base(
            self,
            roots,
            index_path=index_path,
            metadata=metadata,
            semantic=True if repair_like else semantic,
        )

    forced._mmm_small_model_semantic_repair_index = True
    forced.__wrapped__ = base

    class Service:
        index_project_rag = forced

    module = SimpleNamespace(ProductionToolService=Service)
    _install_explicit_semantic_index_policy(module)
    service = Service()
    repair_metadata = {"source_commit": "sha256:x", "license": "project-local"}

    implicit = service.index_project_rag(["src"], metadata=repair_metadata)
    explicit = service.index_project_rag(
        ["src"],
        metadata=repair_metadata,
        semantic=True,
    )

    assert implicit == {"semantic": False}
    assert explicit == {"semantic": True}
    assert calls == [False, True]


def test_strong_lexical_pre_design_result_skips_dense_work() -> None:
    dense_calls: list[str] = []

    def lexical(index_path, query):
        del index_path
        return {
            "status": "searched",
            "query": query,
            "hits": [{"text": "matching source"}],
            "receipt": {
                "result_count": 1,
                "coverage_score": 0.9,
                "relevance_score": 1.0,
            },
        }

    @wraps(lexical)
    def hybrid(index_path, query):
        dense_calls.append(query)
        value = lexical(index_path, query)
        value["retrieval_mode"] = "semantic+rerank"
        return value

    hybrid._mmm_small_model_hybrid_code_rag = True
    hybrid.__wrapped__ = lexical
    module = SimpleNamespace(_search_code_index=hybrid)
    _install_pre_design_rag_cascade(module)

    result = module._search_code_index("rag.db", "find registry code")

    assert not dense_calls
    assert result["dense_work_skipped"] is True
    assert result["retrieval_mode"] == "lexical-strong-no-dense-work"


def test_weak_lexical_pre_design_result_keeps_dense_fallback() -> None:
    dense_calls: list[str] = []

    def lexical(index_path, query):
        del index_path
        return {
            "status": "searched",
            "query": query,
            "hits": [],
            "receipt": {
                "result_count": 0,
                "coverage_score": 0.0,
                "relevance_score": 0.0,
            },
        }

    @wraps(lexical)
    def hybrid(index_path, query):
        dense_calls.append(query)
        return {
            "status": "searched",
            "query": query,
            "hits": [{"text": "dense evidence"}],
            "receipt": {
                "result_count": 1,
                "coverage_score": 0.8,
                "relevance_score": 1.0,
            },
            "retrieval_mode": "lexical+rerank",
        }

    hybrid._mmm_small_model_hybrid_code_rag = True
    hybrid.__wrapped__ = lexical
    module = SimpleNamespace(_search_code_index=hybrid)
    _install_pre_design_rag_cascade(module)

    result = module._search_code_index("rag.db", "find hidden implementation")

    assert dense_calls == ["find hidden implementation"]
    assert result["dense_work_skipped"] is False
    assert result["retrieval_mode"] == "lexical+rerank"
