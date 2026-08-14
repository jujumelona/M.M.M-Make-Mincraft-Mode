from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.model_adapters.base as adapter_base
import minecraft_mod_ai.model_runtime_performance as runtime
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.reranker import RerankerAdapter


class _FakeBatch(dict):
    def to(self, device: str):
        assert device == "cpu"
        return self


class _FakeTokenizer:
    load_count = 0
    instance: "_FakeTokenizer | None" = None

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.load_count += 1
        cls.instance = cls()
        return cls.instance

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(str(item["content"]) for item in messages)

    def __call__(self, rendered, **kwargs):
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["return_tensors"] == "pt"
        batch = list(rendered)
        self.batches.append(batch)
        return _FakeBatch(rendered=batch)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"no": 0, "yes": 1}[token]


def _score_from_rendered(text: str) -> float:
    return float(text.rsplit("SCORE=", 1)[1].split()[0])


class _FakeLogits:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def __getitem__(self, key):
        assert key == (slice(None), -1, [0, 1])
        return self.scores


class _FakeVector:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self) -> list[float]:
        return self.scores


class _FakeProbabilities:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def __getitem__(self, key):
        assert key == (slice(None), 1)
        return _FakeVector(self.scores)


class _FakeModel:
    load_count = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.load_count += 1
        return cls()

    def to(self, device: str):
        assert device == "cpu"
        return self

    def eval(self) -> None:
        return None

    def parameters(self):
        return iter((SimpleNamespace(device="cpu"),))

    def __call__(self, **inputs):
        scores = [_score_from_rendered(text) for text in inputs["rendered"]]
        return SimpleNamespace(logits=_FakeLogits(scores))


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def _cached_score_implementation():
    implementation = RerankerAdapter.score
    while not getattr(implementation, "_mmm_cached_reranker_model", False):
        implementation = implementation.__wrapped__
    return implementation


def test_auto_microbatch_size_uses_live_cpu_and_available_ram(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RERANK_MICROBATCH", raising=False)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(runtime, "_available_memory_bytes", lambda: 5 * runtime._GIB)

    assert runtime._rerank_microbatch_size(20) == 2


def test_explicit_microbatch_override_is_honored_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RERANK_MICROBATCH", "3")
    assert runtime._rerank_microbatch_size(20) == 3

    monkeypatch.setenv("MMM_RERANK_MICROBATCH", "999")
    assert runtime._rerank_microbatch_size(100) == runtime._MAX_RERANK_MICROBATCH

    monkeypatch.setenv("MMM_RERANK_MICROBATCH", "0")
    with pytest.raises(ValueError, match="positive integer"):
        runtime._rerank_microbatch_size(20)


def test_reranker_buckets_by_length_restores_order_and_reuses_one_model(
    monkeypatch,
) -> None:
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=_FakeModel,
        AutoTokenizer=_FakeTokenizer,
    )
    fake_torch = SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        inference_mode=lambda: _InferenceMode(),
        softmax=lambda scores, dim: _FakeProbabilities(scores),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(adapter_base, "require_package", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("MMM_CPU_RETRIEVAL_CACHE", "1")
    monkeypatch.setenv("MMM_RERANK_MICROBATCH", "2")
    monkeypatch.setattr(runtime, "_RERANK_TOKENIZER", None)
    monkeypatch.setattr(runtime, "_RERANK_MODEL", None)
    monkeypatch.setattr(runtime, "_RERANK_KEY", None)
    _FakeTokenizer.load_count = 0
    _FakeModel.load_count = 0

    adapter = RerankerAdapter(
        AdapterConfig(
            role="reranker",
            adapter="reranker",
            model_id="fake/reranker",
            max_context=512,
        )
    )
    documents = [
        "SCORE=0.11 " + ("L" * 80),
        "SCORE=0.22 x",
        "SCORE=0.33 " + ("M" * 30),
        "SCORE=0.44 " + ("S" * 5),
        "SCORE=0.55 " + ("X" * 120),
    ]

    score = _cached_score_implementation()
    values = score(adapter, "query", documents)
    second_values = score(adapter, "query", ["SCORE=0.66 z"])

    assert values == [0.11, 0.22, 0.33, 0.44, 0.55]
    assert second_values == [0.66]
    tokenizer = _FakeTokenizer.instance
    assert tokenizer is not None
    first_call_batches = tokenizer.batches[:3]
    assert [len(batch) for batch in first_call_batches] == [2, 2, 1]
    observed = [
        _score_from_rendered(text)
        for batch in first_call_batches
        for text in batch
    ]
    expected = [
        float(document.split("SCORE=", 1)[1].split()[0])
        for _index, document in sorted(
            enumerate(documents),
            key=lambda item: (len(item[1]), item[0]),
        )
    ]
    assert observed == expected
    assert _FakeTokenizer.load_count == 1
    assert _FakeModel.load_count == 1
