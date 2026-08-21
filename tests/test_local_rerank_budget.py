from __future__ import annotations

from minecraft_mod_ai.retrieval_model_residency import _bounded_rerank_scores


class _Adapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def score(self, query, documents, *, instruction):
        del query, instruction
        values = tuple(documents)
        self.calls.append(values)
        return [float(len(values) - index) for index, _value in enumerate(values)]


def test_local_cpu_reranker_scores_only_bounded_preranked_prefix(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LOCAL_RERANK_DOCUMENTS", raising=False)
    adapter = _Adapter()
    documents = tuple(f"doc-{index}" for index in range(40))

    scores = _bounded_rerank_scores(
        adapter,
        "query",
        documents,
        instruction="rank",
        local_cpu=True,
    )

    assert adapter.calls == [documents[:8]]
    assert len(scores) == len(documents)
    assert max(scores[8:]) < min(scores[:8])


def test_local_reranker_budget_can_be_explicitly_raised(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LOCAL_RERANK_DOCUMENTS", "12")
    adapter = _Adapter()
    documents = tuple(f"doc-{index}" for index in range(20))

    scores = _bounded_rerank_scores(
        adapter,
        "query",
        documents,
        instruction="rank",
        local_cpu=True,
    )

    assert adapter.calls == [documents[:12]]
    assert len(scores) == 20


def test_non_cpu_reranker_keeps_full_candidate_set(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LOCAL_RERANK_DOCUMENTS", raising=False)
    adapter = _Adapter()
    documents = tuple(f"doc-{index}" for index in range(20))

    scores = _bounded_rerank_scores(
        adapter,
        "query",
        documents,
        instruction="rank",
        local_cpu=False,
    )

    assert adapter.calls == [documents]
    assert len(scores) == len(documents)
