from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.model_adapters.transformers_text as text_adapter
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    GenerationRequest,
    ModelBackendError,
    ModelConfigurationError,
)


class _FakeTokenizer:
    instance: _FakeTokenizer | None = None
    token_count = 9
    load_count = 0

    def __init__(self) -> None:
        self.rendered = ""
        self.tokenize_kwargs: dict[str, object] = {}
        self.eos_token_id = 0

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
        self.rendered = "\n".join(str(item["content"]) for item in messages)
        return self.rendered

    def __call__(self, rendered: str, **kwargs):
        assert rendered == self.rendered
        self.tokenize_kwargs = dict(kwargs)
        return {
            "input_ids": _FakeTensor(self.token_count),
        }

    def decode(self, generated, *, skip_special_tokens: bool) -> str:
        assert generated == "generated-token-slice"
        assert skip_special_tokens is True
        return "BOUNDED_PAGE_ACCEPTED"


class _FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)

    def to(self, _device):
        return self


class _UnexpectedModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        raise AssertionError("overflow must fail before loading model weights")


class _FakeOutput:
    def __getitem__(self, key):
        assert key == (0, slice(_FakeTokenizer.token_count, None))
        return "generated-token-slice"


class _FakeModel:
    load_kwargs: dict[str, object] = {}
    load_count = 0

    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        cls.load_count += 1
        cls.load_kwargs = dict(kwargs)
        return cls()

    def parameters(self):
        return iter((SimpleNamespace(device="cpu"),))

    def generate(self, **kwargs):
        assert kwargs["input_ids"].shape == (
            1,
            _FakeTokenizer.token_count,
        )
        return _FakeOutput()


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model,
) -> None:
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=model,
        AutoTokenizer=_FakeTokenizer,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        inference_mode=lambda: _InferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        text_adapter,
        "require_package",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        text_adapter,
        "preflight_cuda",
        lambda _config: None,
    )
    _FakeTokenizer.load_count = 0
    if hasattr(model, "load_count"):
        model.load_count = 0


def test_text_adapter_rejects_context_overflow_without_losing_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "FINAL_REQUIREMENT_MUST_SURVIVE"
    _FakeTokenizer.token_count = 9
    _install_fake_runtime(monkeypatch, model=_UnexpectedModel)

    adapter = text_adapter.TransformersTextAdapter(
        AdapterConfig(
            role="coder",
            adapter="transformers_text",
            model_id="fake/text-model",
            max_context=8,
            max_new_tokens=2,
        )
    )
    with pytest.raises(ModelBackendError) as raised:
        adapter.generate(
            GenerationRequest(
                messages=(
                    {"role": "system", "content": "Keep every requirement."},
                    {"role": "user", "content": f"Long request {sentinel}"},
                )
            )
        )

    assert isinstance(raised.value.cause, ModelConfigurationError)
    assert "9 input + 2 reserved output > max_context=8" in str(raised.value)
    tokenizer = _FakeTokenizer.instance
    assert tokenizer is not None
    assert tokenizer.rendered.endswith(sentinel)
    assert tokenizer.tokenize_kwargs["truncation"] is False
    assert "max_length" not in tokenizer.tokenize_kwargs


def test_text_adapter_accepts_a_bounded_page_at_context_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeTokenizer.token_count = 6
    _install_fake_runtime(monkeypatch, model=_FakeModel)
    adapter = text_adapter.TransformersTextAdapter(
        AdapterConfig(
            role="coder",
            adapter="transformers_text",
            model_id="fake/text-model",
            max_context=8,
            max_new_tokens=2,
        )
    )

    result = adapter.generate(
        GenerationRequest(
            messages=(
                {"role": "user", "content": "one bounded planner page"},
            )
        )
    )

    assert result == "BOUNDED_PAGE_ACCEPTED"
    tokenizer = _FakeTokenizer.instance
    assert tokenizer is not None
    assert tokenizer.tokenize_kwargs == {
        "return_tensors": "pt",
        "truncation": False,
    }
    assert _FakeModel.load_kwargs["dtype"] == "auto"
    assert _FakeModel.load_kwargs["attn_implementation"] == "sdpa"
    assert "torch_dtype" not in _FakeModel.load_kwargs


def test_text_generation_session_reuses_one_backend_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeTokenizer.token_count = 6
    _install_fake_runtime(monkeypatch, model=_FakeModel)
    adapter = text_adapter.TransformersTextAdapter(
        AdapterConfig(
            role="coder",
            adapter="transformers_text",
            model_id="fake/text-model",
            max_context=8,
            max_new_tokens=2,
        )
    )
    request = GenerationRequest(
        messages=(
            {"role": "user", "content": "one bounded planner page"},
        )
    )

    with adapter.generation_session():
        assert adapter.generate(request) == "BOUNDED_PAGE_ACCEPTED"
        assert adapter.generate(request) == "BOUNDED_PAGE_ACCEPTED"
        assert adapter._model is not None
        assert adapter._tokenizer is not None

    assert _FakeTokenizer.load_count == 1
    assert _FakeModel.load_count == 1
    assert adapter._model is None
    assert adapter._tokenizer is None
