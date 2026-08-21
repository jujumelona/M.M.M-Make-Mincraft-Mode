from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai.qwen35_request_policy import install


_SAMPLING_PROFILES = {
    "general_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
    },
    "precise_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "non_thinking": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "reasoning_effort": "none",
    },
}


def _config(role: str = "planner", *, qwen: bool = True):
    if qwen:
        extra = {
            "gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
            "request_policy": "task_aware_sampling",
            "sampling_profiles": _SAMPLING_PROFILES,
        }
    else:
        extra = {"gguf_filename": "gemma-4-12b-it-qat-q4_0.gguf"}
    return SimpleNamespace(
        role=role,
        model_id=(
            "unsloth/Qwen3.5-9B-MTP-GGUF"
            if qwen
            else "google/gemma-4-12B-it-qat-q4_0-gguf"
        ),
        extra=extra,
    )


def _request(*, response_format: str = "text", tools=()):
    return SimpleNamespace(response_format=response_format, tools=tools)


def _hardware():
    def server_payload(adapter, request):
        del request
        return {
            "temperature": 0.0,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget_tokens": 0,
            "model_id": adapter.config.model_id,
        }

    return SimpleNamespace(_server_payload=server_payload)


def _autotune():
    autotune = SimpleNamespace()

    def base_args(_binary, _model_path, _config, _port):
        return ["llama-server", "--reasoning", "auto"]

    def benchmark(binary, model_path, config, request, fingerprint):
        del request, fingerprint
        return autotune._base_args(binary, model_path, config, 18910)

    autotune._base_args = base_args
    autotune._benchmark = benchmark
    autotune._fingerprint = lambda config, binary, model_path: (
        f"{config.model_id}:{binary}:{model_path}"
    )
    return autotune


def _install_isolated():
    hardware = _hardware()
    autotune = _autotune()
    install(autotune, hardware)
    return autotune, hardware


def test_general_thinking_uses_qwen_recommended_sampling() -> None:
    _autotune_module, hardware = _install_isolated()
    adapter = SimpleNamespace(config=_config("planner"))

    payload = hardware._server_payload(adapter, _request())

    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repeat_penalty"] == 1.0
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload
    assert "thinking_budget_tokens" not in payload


def test_precise_coding_keeps_thinking_with_coding_sampling() -> None:
    _autotune_module, hardware = _install_isolated()
    adapter = SimpleNamespace(config=_config("coder_safe"))

    payload = hardware._server_payload(adapter, _request())

    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 0.0
    assert "reasoning_effort" not in payload


def test_host_owned_json_fill_uses_native_non_thinking_mode() -> None:
    _autotune_module, hardware = _install_isolated()
    adapter = SimpleNamespace(config=_config("coder"))

    payload = hardware._server_payload(adapter, _request(response_format="json"))

    assert payload["reasoning_effort"] == "none"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repeat_penalty"] == 1.0
    assert "chat_template_kwargs" not in payload
    assert "thinking_budget_tokens" not in payload


def test_tool_transport_does_not_disable_qwen_reasoning_by_itself() -> None:
    _autotune_module, hardware = _install_isolated()
    adapter = SimpleNamespace(config=_config("planner"))
    tool = {"type": "function", "function": {"name": "lookup"}}

    payload = hardware._server_payload(
        adapter,
        _request(response_format="json", tools=(tool,)),
    )

    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_model_name_without_registry_policy_does_not_activate_qwen_policy() -> None:
    _autotune_module, hardware = _install_isolated()
    config = _config()
    config.extra = {"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"}
    adapter = SimpleNamespace(config=config)

    payload = hardware._server_payload(adapter, _request())

    assert payload["temperature"] == 0.0
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["thinking_budget_tokens"] == 0


def test_non_qwen_payload_is_unchanged() -> None:
    _autotune_module, hardware = _install_isolated()
    adapter = SimpleNamespace(config=_config(qwen=False))

    payload = hardware._server_payload(adapter, _request())

    assert payload["temperature"] == 0.0
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["thinking_budget_tokens"] == 0


def test_qwen_autotune_benchmark_disables_reasoning_only_during_probe() -> None:
    autotune, _hardware_module = _install_isolated()
    config = _config()
    os.environ.pop("MMM_QWEN35_DECODE_BENCHMARK", None)

    outside = autotune._base_args("server", "model.gguf", config, 8910)
    inside = autotune._benchmark("server", "model.gguf", config, object(), "fp")
    after = autotune._base_args("server", "model.gguf", config, 8910)

    assert outside[outside.index("--reasoning") + 1] == "auto"
    assert inside[inside.index("--reasoning") + 1] == "off"
    assert after[after.index("--reasoning") + 1] == "auto"
    assert "MMM_QWEN35_DECODE_BENCHMARK" not in os.environ


def test_qwen_autotune_fingerprint_changes_with_request_policy() -> None:
    autotune, _hardware_module = _install_isolated()
    qwen = _config()
    other = _config(qwen=False)

    qwen_base = f"{qwen.model_id}:server:model.gguf"
    other_base = f"{other.model_id}:server:model.gguf"

    assert autotune._fingerprint(qwen, "server", "model.gguf") != qwen_base
    assert autotune._fingerprint(other, "server", "model.gguf") == other_base
