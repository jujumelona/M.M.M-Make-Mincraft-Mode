from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.model_adapters.transformers_multimodal as multimodal_adapter
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest


class _FakeTensor:
    shape = (1, 3)

    def to(self, _device):
        return self


class _FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_FakeTensor())

    def to(self, _device):
        return self


class _FakeProcessor:
    instance: "_FakeProcessor | None" = None

    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.instance = cls()
        return cls.instance

    def apply_chat_template(self, _messages, **kwargs):
        self.template_kwargs = dict(kwargs)
        return _FakeInputs()

    def batch_decode(self, _generated, *, skip_special_tokens: bool):
        assert skip_special_tokens is True
        return ["{\"game_design\": {}, \"build_slice\": {}}"]


class _FakeModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def parameters(self):
        return iter((SimpleNamespace(device="cpu"),))

    def generate(self, **_kwargs):
        return _FakeOutput()


class _FakeOutput:
    def __getitem__(self, key):
        assert key == (slice(None, None, None), slice(3, None, None))
        return "generated"


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForMultimodalLM=_FakeModel,
            AutoProcessor=_FakeProcessor,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
            inference_mode=lambda: _InferenceMode(),
        ),
    )
    monkeypatch.setattr(
        multimodal_adapter,
        "require_package",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        multimodal_adapter,
        "preflight_cuda",
        lambda _config: None,
    )


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_id: str,
    response_format: str,
) -> dict[str, object]:
    _install_fake_runtime(monkeypatch)
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id=model_id,
            max_context=8,
            max_new_tokens=2,
        )
    )
    result = adapter.generate(
        GenerationRequest(
            messages=({"role": "user", "content": "Return JSON."},),
            response_format=response_format,
        )
    )
    assert result.startswith("{")
    processor = _FakeProcessor.instance
    assert processor is not None
    return processor.template_kwargs


def test_qwen35_json_generation_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _generate(
        monkeypatch,
        model_id="Qwen/Qwen3.5-4B",
        response_format="json",
    )

    assert kwargs["enable_thinking"] is False


def test_other_multimodal_models_do_not_receive_qwen_template_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _generate(
        monkeypatch,
        model_id="example/vision-model",
        response_format="json",
    )

    assert "enable_thinking" not in kwargs


def test_qwen35_text_generation_keeps_its_normal_template_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _generate(
        monkeypatch,
        model_id="Qwen/Qwen3.5-4B",
        response_format="text",
    )

    assert "enable_thinking" not in kwargs
