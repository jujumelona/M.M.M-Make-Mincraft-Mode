"""Regression: deterministic Java import loops cannot silently restart the whole build."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.construction_file_prompt import generate_file_text
from minecraft_mod_ai.llama_completion_liveness_contract import LlamaDegenerateOutputError
from minecraft_mod_ai import llama_server_hardware_policy as hardware
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig, GenerationRequest, GenerationResponse, ModelBackendError,
)
from minecraft_mod_ai.model_adapters import llama_cpp_adapter as adapter_module


class _RepetitionThenSuccessRouter:
    def __init__(self):
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, [dict(x) for x in messages], kwargs))
        if len(self.calls) == 1:
            raise ModelBackendError(
                role="coder", model_id="local",
                cause=LlamaDegenerateOutputError(
                    "LLAMA_DEGENERATE_REPEATED_OUTPUT: repeated imports"
                ),
            )
        return "package test;\npublic class Example {}\n"


def test_repeated_import_loop_switches_protocol_without_dropping_original_task(monkeypatch):
    router = _RepetitionThenSuccessRouter()
    task = {"original_feature_goal": "Fight aliens", "inspected_target_api": {"class_members": []}}
    messages = [
        {"role": "system", "content": "Use exactly the specified package and public class; include all necessary imports, members."},
        {"role": "user", "content": json.dumps(task)},
    ]
    result = generate_file_text(router, messages, target_path="src/Example.java")
    assert result.startswith("package test;")
    assert len(router.calls) == 2
    assert "ZERO import lines" in router.calls[1][1][0]["content"]
    assert "do not emit import declarations" in router.calls[1][1][0]["content"]
    assert json.loads(router.calls[1][1][1]["content"])["original_feature_goal"] == "Fight aliens"
    assert router.calls[1][2]["force_non_thinking"] is True


def test_non_decoder_failures_are_not_hidden_by_source_retries():
    class BrokenRouter:
        def __init__(self):
            self.calls = 0

        def generate_text(self, *_args, **_kwargs):
            self.calls += 1
            raise ValueError("COMPILER_OR_BACKEND_CONFIGURATION_ERROR")

    router = BrokenRouter()
    with pytest.raises(ValueError, match="COMPILER_OR_BACKEND_CONFIGURATION_ERROR"):
        generate_file_text(router, [{"role": "system", "content": "test"}], target_path="src/X.java")
    assert router.calls == 1


def test_repetition_recovery_payload_changes_greedy_sampling_but_not_thinking():
    config = AdapterConfig(
        role="coder", adapter="llama_cpp", model_id="local",
        max_new_tokens=1024, extra={"runtime_contract": "mimo"},
    )
    adapter = SimpleNamespace(config=config)
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create a Java file"},),
        metadata={"mmm_force_non_thinking": True},
    )
    normal = hardware._server_payload(adapter, request)
    recovered = hardware._server_payload(
        adapter,
        GenerationRequest(
            messages=request.messages,
            metadata={"mmm_force_non_thinking": True,
                      "mmm_decode_repetition_recovery": True},
        ),
    )
    assert normal["temperature"] == 0.0
    assert recovered["temperature"] > 0
    assert recovered["repeat_penalty"] > 1
    assert recovered["chat_template_kwargs"] == {"enable_thinking": False}
    assert recovered["max_tokens"] == normal["max_tokens"]


def test_adapter_resamples_coder_text_once_and_preserves_original_request(monkeypatch):
    config = AdapterConfig(
        role="coder", adapter="llama_cpp", model_id="local",
        extra={"runtime_contract": "mimo"},
    )
    adapter = adapter_module.LlamaCppAdapter(config)
    monkeypatch.setattr(adapter, "_server_url", lambda _request: "http://localhost/v1")
    calls = []

    def decode(_adapter, _url, request):
        calls.append(request)
        if len(calls) == 1:
            raise LlamaDegenerateOutputError("LLAMA_DEGENERATE_REPEATED_OUTPUT")
        return GenerationResponse(content="package test; class Example {}")

    monkeypatch.setattr(adapter_module, "_generate_one_turn", decode)
    original = GenerationRequest(
        messages=({"role": "user", "content": "Java source"},),
        metadata={"mmm_force_non_thinking": True},
    )
    result = adapter.generate_turn(original)
    assert result.content.startswith("package test;")
    assert len(calls) == 2
    assert not original.metadata.get("mmm_decode_repetition_recovery")
    assert calls[1].metadata["mmm_decode_repetition_recovery"] is True


def test_adapter_does_not_resample_structured_turns(monkeypatch):
    adapter = adapter_module.LlamaCppAdapter(
        AdapterConfig(role="coder", adapter="llama_cpp", model_id="local")
    )
    monkeypatch.setattr(adapter, "_server_url", lambda _request: "http://localhost/v1")
    calls = []

    def broken(*_args):
        calls.append(1)
        raise LlamaDegenerateOutputError("LLAMA_DEGENERATE_REPEATED_OUTPUT")

    monkeypatch.setattr(adapter_module, "_generate_one_turn", broken)
    with pytest.raises(ModelBackendError, match="LLAMA_DEGENERATE_REPEATED_OUTPUT"):
        adapter.generate_turn(GenerationRequest(
            messages=({"role": "user", "content": "{}"},),
            response_format="json",
            metadata={"mmm_force_non_thinking": True},
        ))
    assert len(calls) == 1
