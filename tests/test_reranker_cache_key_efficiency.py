from minecraft_mod_ai.model_adapters import reranker
from minecraft_mod_ai.model_adapters.base import AdapterConfig


def _adapter() -> reranker.RerankerAdapter:
    return reranker.RerankerAdapter(
        AdapterConfig(
            role="retrieval_reranker",
            adapter="reranker",
            model_id="test-reranker-model",
            max_context=1024,
            extra={"device": "cpu"},
        )
    )


def test_score_cache_keys_hash_request_once_and_each_document_once(monkeypatch) -> None:
    request_calls: list[tuple[str, str]] = []
    document_calls: list[str] = []

    def request_digest(query: str, instruction: str) -> str:
        request_calls.append((query, instruction))
        return "request-digest"

    def text_digest(document: str) -> str:
        document_calls.append(document)
        return f"document:{document}"

    monkeypatch.setattr(reranker, "_request_digest", request_digest)
    monkeypatch.setattr(reranker, "_text_digest", text_digest)

    keys = _adapter()._score_cache_keys(
        "query",
        "instruction",
        ["alpha", "alpha", "beta"],
    )

    assert request_calls == [("query", "instruction")]
    assert document_calls == ["alpha", "beta"]
    assert keys["alpha"][-2:] == ("request-digest", "document:alpha")
    assert keys["beta"][-2:] == ("request-digest", "document:beta")


def test_score_cache_hit_reuses_precomputed_commitments(monkeypatch) -> None:
    request_calls: list[tuple[str, str]] = []
    document_calls: list[str] = []

    def request_digest(query: str, instruction: str) -> str:
        request_calls.append((query, instruction))
        return "request-digest"

    def text_digest(document: str) -> str:
        document_calls.append(document)
        return f"document:{document}"

    monkeypatch.setattr(reranker, "_request_digest", request_digest)
    monkeypatch.setattr(reranker, "_text_digest", text_digest)
    adapter = _adapter()
    instruction = "instruction"
    keys = adapter._score_cache_keys("query", instruction, ["alpha", "beta"])
    request_calls.clear()
    document_calls.clear()

    with reranker._SCORE_CACHE_LOCK:
        reranker._SCORE_CACHE[keys["alpha"]] = 0.75
        reranker._SCORE_CACHE[keys["beta"]] = 0.25

    try:
        assert adapter.score(
            "query",
            ["alpha", "alpha", "beta"],
            instruction=instruction,
        ) == [0.75, 0.75, 0.25]
        assert request_calls == [("query", instruction)]
        assert document_calls == ["alpha", "beta"]
    finally:
        with reranker._SCORE_CACHE_LOCK:
            for key in keys.values():
                reranker._SCORE_CACHE.pop(key, None)
