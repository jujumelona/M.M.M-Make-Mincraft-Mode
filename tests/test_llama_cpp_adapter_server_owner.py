from __future__ import annotations

from minecraft_mod_ai import llama_server_autotune
from minecraft_mod_ai.model_adapters import llama_cpp_adapter as adapter_module
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    GenerationRequest,
    GenerationResponse,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def test_adapter_does_not_bypass_server_owner_when_env_url_exists(monkeypatch) -> None:
    config = AdapterConfig(
        role="visual_critic",
        adapter="llama_cpp",
        model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        extra={
            "gguf_filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
        },
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect this image."},),
    )
    seen: list[tuple[AdapterConfig, GenerationRequest]] = []

    def ensure_tuned_server(
        seen_config: AdapterConfig,
        seen_request: GenerationRequest,
    ) -> str:
        seen.append((seen_config, seen_request))
        return "http://127.0.0.1:8920/v1"

    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(
        llama_server_autotune,
        "ensure_tuned_server",
        ensure_tuned_server,
    )

    adapter = LlamaCppAdapter(config)
    assert adapter._server_url(request) == "http://127.0.0.1:8920/v1"
    assert seen == [(config, request)]


def test_adapter_replays_one_turn_after_managed_transport_recovery(monkeypatch) -> None:
    config = AdapterConfig(
        role="coder",
        adapter="llama_cpp",
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create one source file."},),
    )
    adapter = LlamaCppAdapter(config)
    old_url = "http://127.0.0.1:8910/v1"
    new_url = "http://127.0.0.1:8911/v1"
    generated_urls: list[str] = []
    recovery_calls: list[tuple[AdapterConfig, GenerationRequest, str]] = []

    monkeypatch.setattr(adapter, "_server_url", lambda _request: old_url)

    def generate_once(_adapter, server_url, _request):
        generated_urls.append(server_url)
        if len(generated_urls) == 1:
            raise adapter_module.httpx.ConnectError("connection refused")
        return GenerationResponse(content="recovered")

    monkeypatch.setattr(adapter_module, "_generate_one_turn", generate_once)

    def recover(seen_config, seen_request, *, failed_url):
        recovery_calls.append((seen_config, seen_request, failed_url))
        return new_url

    monkeypatch.setattr(llama_server_autotune, "recover_managed_server", recover)

    result = adapter.generate_turn(request)

    assert result.content == "recovered"
    assert generated_urls == [old_url, new_url]
    assert recovery_calls == [(config, request, old_url)]
