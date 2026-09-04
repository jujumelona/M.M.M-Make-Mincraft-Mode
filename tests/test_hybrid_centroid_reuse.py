from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import minecraft_mod_ai.small_model_hybrid_search_contract as hybrid


class _Service:
    profile = "test"
    router = object()

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve(self, path: str, allow_root: bool = False) -> Path:
        del allow_root
        return self.root / path

    def search_code_rag(
        self,
        query: str,
        *,
        index_path: str = "rag/project-index.json",
        limit: int = 8,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del index_path, limit, semantic, rerank, required_metadata
        return {
            "query": query,
            "hits": [{"text": "Alpha Beta Gamma"}],
            "receipt": {
                "result_count": 1,
                "coverage_score": 0.1,
                "relevance_score": 0.1,
            },
        }


def test_centroid_text_fallback_reuses_direct_q1(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def adapt(router, query, texts):
        nonlocal calls
        del router, query
        assert texts == ["Alpha Beta Gamma"]
        calls += 1
        return [1.0, 0.0]

    monkeypatch.setattr(hybrid, "extract_hit_texts", lambda result: ["Alpha Beta Gamma"])
    monkeypatch.setattr(hybrid, "adapt_query_vector", adapt)
    monkeypatch.setattr(
        hybrid,
        "direct_centroid_vector_search",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hybrid,
        "_embedding_rows",
        lambda router, tokens: [[1.0, 0.0] for _token in tokens],
    )

    module = SimpleNamespace(ProductionToolService=_Service)
    hybrid.install(module)

    service = _Service(tmp_path)
    result = service.search_code_rag("ordinary behavior lookup")

    assert result["task_route"] == "semantic"
    assert calls == 1
