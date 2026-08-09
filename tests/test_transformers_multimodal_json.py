from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.model_adapters.base as adapter_base
import minecraft_mod_ai.model_adapters.transformers_multimodal as multimodal_adapter
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.base import (
    ModelBackendError,
    ModelConfigurationError,
)


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
    load_count = 0

    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] = {}
        self.messages: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.load_count += 1
        cls.instance = cls()
        return cls.instance

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = dict(kwargs)
        self.messages = [dict(message) for message in messages]
        return _FakeInputs()

    def batch_decode(self, _generated, *, skip_special_tokens: bool):
        assert skip_special_tokens is True
        return ["{\"game_design\": {}, \"build_slice\": {}}"]


class _FakeModel:
    load_count = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.load_count += 1
        return cls()

    def parameters(self):
        return iter((SimpleNamespace(device="cpu"),))

    def generate(self, **_kwargs):
        return _FakeOutput()


class _UnexpectedModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        raise AssertionError("overflow must fail before loading model weights")


class _FakeOutput:
    def __getitem__(self, key):
        assert key == (slice(None, None, None), slice(3, None, None))
        return "generated"


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model=_FakeModel,
) -> None:
    _FakeProcessor.instance = None
    _FakeProcessor.load_count = 0
    if hasattr(model, "load_count"):
        model.load_count = 0
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForMultimodalLM=model,
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


def test_multimodal_context_overflow_fails_before_loading_model_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "FINAL_REQUIREMENT_MUST_SURVIVE"
    monkeypatch.setattr(_FakeTensor, "shape", (1, 9))
    _install_fake_runtime(monkeypatch, model=_UnexpectedModel)
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            max_context=10,
            max_new_tokens=2,
        )
    )

    with pytest.raises(ModelBackendError) as raised:
        adapter.generate(
            GenerationRequest(
                messages=(
                    {
                        "role": "user",
                        "content": f"Long request {sentinel}",
                    },
                ),
                response_format="json",
            )
        )

    assert isinstance(raised.value.cause, ModelConfigurationError)
    assert "9 input + 2 reserved output > max_context=10" in str(raised.value)
    processor = _FakeProcessor.instance
    assert processor is not None
    assert processor.messages[-1]["content"].endswith(sentinel)
    assert "truncation" not in processor.template_kwargs


def test_multimodal_generation_session_loads_once_and_releases_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    releases: list[str] = []
    monkeypatch.setattr(
        adapter_base,
        "_release_cuda",
        lambda: releases.append("released"),
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-9B",
            max_context=8,
            max_new_tokens=2,
        )
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Return one page."},),
        response_format="json",
    )

    with adapter.generation_session():
        assert adapter.generate(request).startswith("{")
        assert adapter.generate(request).startswith("{")
        assert adapter._processor is not None
        assert adapter._model is not None
        assert releases == []

    assert _FakeProcessor.load_count == 1
    assert _FakeModel.load_count == 1
    assert adapter._processor is None
    assert adapter._model is None
    assert releases == ["released"]


def test_direct_multimodal_generate_keeps_auto_release_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    releases: list[str] = []
    monkeypatch.setattr(
        adapter_base,
        "_release_cuda",
        lambda: releases.append("released"),
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-9B",
            max_context=8,
            max_new_tokens=2,
        )
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Return one page."},),
        response_format="json",
    )

    adapter.generate(request)
    adapter.generate(request)

    assert _FakeProcessor.load_count == 2
    assert _FakeModel.load_count == 2
    assert adapter._processor is None
    assert adapter._model is None
    assert releases == ["released", "released"]


class _FailingModel(_FakeModel):
    def generate(self, **_kwargs):
        raise RuntimeError("generation failed")


def test_multimodal_generation_session_releases_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, model=_FailingModel)
    releases: list[str] = []
    monkeypatch.setattr(
        adapter_base,
        "_release_cuda",
        lambda: releases.append("released"),
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-9B",
            max_context=8,
            max_new_tokens=2,
        )
    )

    with pytest.raises(ModelBackendError, match="generation failed"):
        with adapter.generation_session():
            adapter.generate(
                GenerationRequest(
                    messages=(
                        {"role": "user", "content": "Return one page."},
                    ),
                    response_format="json",
                )
            )

    assert _FakeProcessor.load_count == 1
    assert _FailingModel.load_count == 1
    assert adapter._processor is None
    assert adapter._model is None
    assert releases == ["released"]
