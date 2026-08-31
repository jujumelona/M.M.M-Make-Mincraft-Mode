from __future__ import annotations

from minecraft_mod_ai.llama_finish_reason_contract import (
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
)
from minecraft_mod_ai.model_adapters import llama_cpp_adapter as adapter_module
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def _adapter(*, extra=None) -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="coder",
            adapter="llama_cpp",
            model_id="test/model",
            max_new_tokens=8192,
            extra=extra or {},
        )
    )


def _boundary(content: str) -> LlamaCompletionBoundaryError:
    return LlamaCompletionBoundaryError(
        "typed boundary",
        kind=OUTPUT_EXHAUSTED,
        partial_message={"role": "assistant", "content": content},
        prompt_tokens=100,
        completion_tokens=8192,
        max_tokens=8192,
    )


def _qwen_nonthinking_extra() -> dict[str, object]:
    return {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
    }


def test_warm_repeated_prefill_reuses_one_managed_server_calibration(monkeypatch) -> None:
    prefix = "<think>\n\n</think>\n\n"
    calibration_calls = 0
    outcomes = []
    for _ in range(3):
        outcomes.extend(
            [
                _boundary("A"),
                {"role": "assistant", "content": prefix + "B"},
            ]
        )

    def complete(_url, _payload):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def calibrate(_url, _payload):
        nonlocal calibration_calls
        calibration_calls += 1
        return prefix

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        calibrate,
    )
    monkeypatch.setattr(
        adapter_module,
        "_assistant_prefill_server_identity",
        lambda _url: "managed:test-model:pid=123",
    )

    adapter = _adapter(extra=_qwen_nonthinking_extra())
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    results = [
        adapter_module._completion_message_with_prefill(
            adapter, "http://local", payload
        )["content"]
        for _ in range(3)
    ]

    assert results == ["AB", "AB", "AB"]
    assert calibration_calls == 1
    assert not outcomes


def test_prefill_cache_is_scoped_to_wire_template_shape(monkeypatch) -> None:
    calibration_calls = 0

    def calibrate(_url, _payload):
        nonlocal calibration_calls
        calibration_calls += 1
        return "prefix"

    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        calibrate,
    )
    monkeypatch.setattr(
        adapter_module,
        "_assistant_prefill_server_identity",
        lambda _url: "managed:test-model:pid=123",
    )
    adapter = _adapter()
    first = {"model": "local", "chat_template_kwargs": {"enable_thinking": False}}
    second = {"model": "local", "chat_template_kwargs": {"enable_thinking": True}}

    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://local", first
    )
    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://local", first
    )
    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://local", second
    )

    assert calibration_calls == 2


def test_prefill_cache_is_scoped_to_managed_process_generation(monkeypatch) -> None:
    calibration_calls = 0
    identity = {"value": "managed:test-model:pid=123"}

    def calibrate(_url, _payload):
        nonlocal calibration_calls
        calibration_calls += 1
        return "prefix"

    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        calibrate,
    )
    monkeypatch.setattr(
        adapter_module,
        "_assistant_prefill_server_identity",
        lambda _url: identity["value"],
    )
    adapter = _adapter()
    payload = {"model": "local", "chat_template_kwargs": {"enable_thinking": False}}

    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://local", payload
    )
    identity["value"] = "managed:test-model:pid=456"
    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://local", payload
    )

    assert calibration_calls == 2


def test_external_server_prefill_is_recalibrated_instead_of_cached(monkeypatch) -> None:
    calibration_calls = 0

    def calibrate(_url, _payload):
        nonlocal calibration_calls
        calibration_calls += 1
        return "prefix"

    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        calibrate,
    )
    monkeypatch.setattr(
        adapter_module,
        "_assistant_prefill_server_identity",
        lambda _url: "",
    )
    adapter = _adapter()
    payload = {"model": "local", "chat_template_kwargs": {"enable_thinking": False}}

    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://external", payload
    )
    adapter_module._cached_assistant_prefill_generation_prompt(
        adapter, "http://external", payload
    )

    assert calibration_calls == 2


def test_cached_prefix_mismatch_recalibrates_once_and_preserves_output(monkeypatch) -> None:
    old_prefix = "<old-template>"
    new_prefix = "<new-template>"
    calibration_prefixes = [old_prefix, new_prefix]
    outcomes = [
        _boundary("A"),
        {"role": "assistant", "content": old_prefix + "B"},
        _boundary("C"),
        {"role": "assistant", "content": new_prefix + "D"},
    ]

    def complete(_url, _payload):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def calibrate(_url, _payload):
        return calibration_prefixes.pop(0)

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        calibrate,
    )
    monkeypatch.setattr(
        adapter_module,
        "_assistant_prefill_server_identity",
        lambda _url: "managed:test-model:pid=123",
    )

    adapter = _adapter(extra=_qwen_nonthinking_extra())
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    first = adapter_module._completion_message_with_prefill(
        adapter, "http://local", payload
    )
    second = adapter_module._completion_message_with_prefill(
        adapter, "http://local", payload
    )

    assert first["content"] == "AB"
    assert second["content"] == "CD"
    assert calibration_prefixes == []
    assert outcomes == []


def test_concurrent_managed_prefill_calibration_is_single_flight(monkeypatch) -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    calibration_calls = 0
    start = threading.Barrier(8)

    def calibrate(_url, _payload):
        nonlocal calibration_calls
        calibration_calls += 1
        time.sleep(0.02)
        return "prefix"

    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        calibrate,
    )
    monkeypatch.setattr(
        adapter_module,
        "_assistant_prefill_server_identity",
        lambda _url: "managed:test-model:pid=123",
    )
    adapter = _adapter()
    payload = {"model": "local", "chat_template_kwargs": {"enable_thinking": False}}

    def invoke(_index):
        start.wait()
        return adapter_module._cached_assistant_prefill_generation_prompt(
            adapter, "http://local", payload
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(8)))

    assert results == ["prefix"] * 8
    assert calibration_calls == 1
