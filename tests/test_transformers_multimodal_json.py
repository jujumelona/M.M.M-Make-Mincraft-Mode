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


class _CapturingLoadModel(_FakeModel):
    load_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        cls.load_count += 1
        cls.load_kwargs = dict(kwargs)
        return cls()


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
    monkeypatch.setattr(
        multimodal_adapter,
        "_qwen35_fast_path",
        lambda: None,
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


def test_qwen35_forces_memory_efficient_sdpa_and_current_dtype_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long Qwen page must not materialize eager [heads, tokens, tokens] scores."""

    _CapturingLoadModel.load_kwargs = {}
    _install_fake_runtime(monkeypatch, model=_CapturingLoadModel)
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            torch_dtype="float16",
            max_context=8,
            max_new_tokens=2,
        )
    )

    adapter.generate(
        GenerationRequest(
            messages=({"role": "user", "content": "Return one bounded page."},),
            response_format="json",
        )
    )

    assert _CapturingLoadModel.load_kwargs["attn_implementation"] == "sdpa"
    assert _CapturingLoadModel.load_kwargs["dtype"] == "float16"
    assert "torch_dtype" not in _CapturingLoadModel.load_kwargs


def test_qwen35_rebinds_stale_transformers_fast_path_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def causal_fn():
        return None

    def causal_update():
        return None

    def chunk_rule():
        return None

    def recurrent_rule():
        return None

    class FusedNorm:
        pass

    modeling = SimpleNamespace(
        causal_conv1d_fn=None,
        causal_conv1d_update=None,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
        torch_chunk_gated_delta_rule=None,
        torch_recurrent_gated_delta_rule=None,
        FusedRMSNormGated=None,
        is_fast_path_available=False,
    )
    modules = {
        "transformers.models.qwen3_5.modeling_qwen3_5": modeling,
        "causal_conv1d": SimpleNamespace(
            causal_conv1d_fn=causal_fn,
            causal_conv1d_update=causal_update,
        ),
        "fla.ops.gated_delta_rule": SimpleNamespace(
            chunk_gated_delta_rule=chunk_rule,
            fused_recurrent_gated_delta_rule=recurrent_rule,
        ),
        "fla.modules": SimpleNamespace(FusedRMSNormGated=FusedNorm),
    }
    monkeypatch.setattr(
        multimodal_adapter.importlib,
        "import_module",
        lambda name: modules[name],
    )

    multimodal_adapter._qwen35_fast_path()

    assert modeling.causal_conv1d_fn is causal_fn
    assert modeling.causal_conv1d_update is causal_update
    assert modeling.chunk_gated_delta_rule is chunk_rule
    assert modeling.fused_recurrent_gated_delta_rule is recurrent_rule
    # Keep the explicit torch reference functions as fallbacks. Qwen layers
    # capture the four verified fast globals above during construction.
    assert modeling.torch_chunk_gated_delta_rule is None
    assert modeling.torch_recurrent_gated_delta_rule is None
    assert modeling.FusedRMSNormGated is FusedNorm
    assert modeling.is_fast_path_available is True


def test_qwen35_runtime_versions_are_fail_closed_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    requirements: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        multimodal_adapter,
        "require_package",
        lambda distribution, **kwargs: requirements.append(
            (distribution, dict(kwargs))
        ),
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            max_context=8,
            max_new_tokens=2,
        )
    )

    adapter.generate(
        GenerationRequest(
            messages=({"role": "user", "content": "Return JSON."},),
            response_format="json",
        )
    )

    assert (
        "transformers",
        {"minimum": "5.14.1", "maximum_exclusive": "5.15"},
    ) in requirements
    assert (
        "flash-linear-attention",
        {"minimum": "0.5.1", "maximum_exclusive": "0.6"},
    ) in requirements
    assert (
        "causal-conv1d",
        {"minimum": "1.4.0"},
    ) in requirements


@pytest.mark.parametrize("installed", ["5.13.9", "5.15.0"])
def test_transformers_version_outside_verified_minor_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    installed: str,
) -> None:
    monkeypatch.setattr(
        adapter_base.importlib.metadata,
        "version",
        lambda _distribution: installed,
    )

    with pytest.raises(ModelConfigurationError):
        adapter_base.require_package(
            "transformers",
            minimum="5.14.1",
            maximum_exclusive="5.15",
        )


def test_qwen35_missing_fast_path_fails_before_processor_or_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)

    def fail_fast_path() -> None:
        raise ModelConfigurationError("Qwen3.5 fast CUDA kernels are unavailable")

    monkeypatch.setattr(
        multimodal_adapter,
        "_qwen35_fast_path",
        fail_fast_path,
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            max_context=8,
            max_new_tokens=2,
        )
    )

    with pytest.raises(ModelBackendError, match="fast CUDA kernels are unavailable"):
        adapter.generate(
            GenerationRequest(
                messages=({"role": "user", "content": "Return JSON."},),
                response_format="json",
            )
        )

    assert _FakeProcessor.load_count == 0
    assert _FakeModel.load_count == 0


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


def test_multimodal_page_input_budget_fails_before_loading_model_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_FakeTensor, "shape", (1, 5))
    _install_fake_runtime(monkeypatch, model=_UnexpectedModel)
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            max_context=16,
            max_input_tokens=4,
            max_new_tokens=2,
        )
    )

    with pytest.raises(ModelBackendError) as raised:
        adapter.generate(
            GenerationRequest(
                messages=({"role": "user", "content": "One page."},),
                response_format="json",
            )
        )

    assert isinstance(raised.value.cause, ModelConfigurationError)
    assert "5 input tokens > max_input_tokens=4" in str(raised.value)
    assert "additional pages" in str(raised.value)


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


class _FakeCudaOOM(RuntimeError):
    pass


class _OomModel(_FakeModel):
    def generate(self, **_kwargs):
        raise _FakeCudaOOM("tried to allocate a large attention tensor")


class _OomLoadingModel(_FakeModel):
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        raise _FakeCudaOOM("tried to allocate while loading checkpoint")


def test_multimodal_oom_reports_page_token_counts_and_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, model=_OomModel)
    fake_torch = sys.modules["torch"]
    monkeypatch.setattr(
        fake_torch,
        "OutOfMemoryError",
        _FakeCudaOOM,
        raising=False,
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            max_context=8,
            max_input_tokens=4,
            max_new_tokens=2,
        )
    )

    with pytest.raises(ModelBackendError) as raised:
        adapter.generate(
            GenerationRequest(
                messages=({"role": "user", "content": "Return JSON."},),
                response_format="json",
            )
        )

    assert "input_tokens=3" in str(raised.value)
    assert "max_new_tokens=2" in str(raised.value)
    assert "max_input_tokens=4" in str(raised.value)
    assert "attention_backend=sdpa" in str(raised.value)
    assert "phase=generate" in str(raised.value)


def test_multimodal_model_load_oom_reports_the_actual_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, model=_OomLoadingModel)
    fake_torch = sys.modules["torch"]
    monkeypatch.setattr(
        fake_torch,
        "OutOfMemoryError",
        _FakeCudaOOM,
        raising=False,
    )
    adapter = multimodal_adapter.TransformersMultimodalAdapter(
        AdapterConfig(
            role="planner",
            adapter="transformers_multimodal",
            model_id="Qwen/Qwen3.5-4B",
            max_context=8,
            max_input_tokens=4,
            max_new_tokens=2,
        )
    )

    with pytest.raises(ModelBackendError) as raised:
        adapter.generate(
            GenerationRequest(
                messages=({"role": "user", "content": "Return JSON."},),
                response_format="json",
            )
        )

    assert "phase=model_load" in str(raised.value)
    assert "during multimodal phase=generate" not in str(raised.value)


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
