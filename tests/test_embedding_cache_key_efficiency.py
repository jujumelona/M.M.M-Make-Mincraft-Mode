from minecraft_mod_ai.model_adapters import embedding
from minecraft_mod_ai.model_adapters.base import AdapterConfig


def _adapter() -> embedding.EmbeddingAdapter:
    return embedding.EmbeddingAdapter(
        AdapterConfig(
            role="retrieval_embedding",
            adapter="embedding",
            model_id="test-embedding-model",
            max_context=1024,
            extra={"device": "cpu", "dimensions": 8},
        )
    )


def test_vector_cache_keys_hash_each_unique_text_once(monkeypatch) -> None:
    calls: list[str] = []

    def digest(text: str) -> str:
        calls.append(text)
        return f"digest:{text}"

    monkeypatch.setattr(embedding, "_text_digest", digest)
    keys = _adapter()._vector_cache_keys(["alpha", "alpha", "beta"], 8)

    assert calls == ["alpha", "beta"]
    assert keys["alpha"][-1] == "digest:alpha"
    assert keys["beta"][-1] == "digest:beta"


def test_embed_cache_hit_reuses_precomputed_keys(monkeypatch) -> None:
    calls: list[str] = []

    def digest(text: str) -> str:
        calls.append(text)
        return f"digest:{text}"

    monkeypatch.setattr(embedding, "_text_digest", digest)
    adapter = _adapter()
    keys = adapter._vector_cache_keys(["alpha", "beta"], 8)
    calls.clear()

    with embedding._VECTOR_CACHE_LOCK:
        for text, vector in {
            "alpha": (1.0, 0.0),
            "beta": (0.0, 1.0),
        }.items():
            embedding._VECTOR_CACHE[keys[text]] = vector

    try:
        assert adapter.embed(["alpha", "alpha", "beta"]) == [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        assert calls == ["alpha", "beta"]
    finally:
        with embedding._VECTOR_CACHE_LOCK:
            for key in keys.values():
                embedding._VECTOR_CACHE.pop(key, None)
