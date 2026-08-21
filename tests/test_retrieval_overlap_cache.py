from __future__ import annotations

import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters import embedding as embedding_module
from minecraft_mod_ai.model_adapters import reranker as reranker_module


def _config(*, adapter: str, dimensions: int = 2) -> AdapterConfig:
    return AdapterConfig(
        role="test",
        adapter=adapter,
        model_id="test/model",
        torch_dtype="float32",
        max_context=128,
        extra={"device": "cpu", "dimensions": dimensions},
    )


def test_embedding_cache_reuses_overlapping_texts_and_deduplicates_one_batch(monkeypatch) -> None:
    monkeypatch.setenv("MMM_CPU_RETRIEVAL_CACHE", "1")
    embedding_module._VECTOR_CACHE.clear()
    calls: list[list[str]] = []

    class FakeModel:
        def encode(self, texts, **_kwargs):
            values = list(texts)
            calls.append(values)
            return [[float(len(text)), float(index)] for index, text in enumerate(values)]

    adapter = embedding_module.EmbeddingAdapter(_config(adapter="embedding"))
    backend = SimpleNamespace(model=FakeModel(), lock=threading.RLock())
    monkeypatch.setattr(adapter, "_ensure_backend", lambda: backend)

    first = adapter.embed(["alpha", "beta"])
    second = adapter.embed(["beta", "gamma", "beta"])

    assert calls == [["alpha", "beta"], ["gamma"]]
    assert second[0] == first[1]
    assert second[2] == first[1]


def test_reranker_cache_scores_only_new_documents_in_overlapping_batches(monkeypatch) -> None:
    monkeypatch.setenv("MMM_CPU_RETRIEVAL_CACHE", "1")
    reranker_module._SCORE_CACHE.clear()
    model_calls: list[int] = []

    class FakeInputs(dict):
        def __init__(self, size: int):
            super().__init__(input_ids=SimpleNamespace(size=size))

        def to(self, _device):
            return self

    class FakeTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return str(messages[-1]["content"])

        def convert_tokens_to_ids(self, token):
            return 1 if token == "yes" else 0

        def __call__(self, texts, **_kwargs):
            return FakeInputs(len(texts))

    class FakeLogits:
        def __init__(self, size: int):
            self.size = size

        def __getitem__(self, _key):
            return self

    class FakeValues:
        def __init__(self, size: int):
            self.size = size

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [0.1 * (index + 1) for index in range(self.size)]

    class FakeProbabilities:
        def __init__(self, size: int):
            self.size = size

        def __getitem__(self, _key):
            return FakeValues(self.size)

    class FakeModel:
        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def __call__(self, **kwargs):
            size = kwargs["input_ids"].size
            model_calls.append(size)
            return SimpleNamespace(logits=FakeLogits(size))

    fake_torch = SimpleNamespace(
        inference_mode=lambda: nullcontext(),
        softmax=lambda logits, dim: FakeProbabilities(logits.size),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(reranker_module, "_rerank_microbatch_size", lambda _size: 16)
    monkeypatch.setattr(
        reranker_module,
        "_length_bucketed_batches",
        lambda values, _batch_size: [list(enumerate(values))],
    )

    adapter = reranker_module.RerankerAdapter(_config(adapter="reranker"))
    backend = SimpleNamespace(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        lock=threading.RLock(),
    )
    monkeypatch.setattr(adapter, "_ensure_backend", lambda: backend)

    first = adapter.score("query", ["alpha", "beta"])
    second = adapter.score("query", ["beta", "gamma", "beta"])

    assert model_calls == [2, 1]
    assert second[0] == first[1]
    assert second[2] == first[1]
