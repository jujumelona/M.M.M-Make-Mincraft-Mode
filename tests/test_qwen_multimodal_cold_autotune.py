from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai.model_adapters.base import GenerationRequest


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        extra={
            "gguf_filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
        },
    )


def test_cold_media_benchmark_runs_without_media_scope(monkeypatch) -> None:
    seen: list[str] = []

    def benchmark(_binary, _model_path, _config, _request, _fingerprint):
        seen.append(os.environ.get(multimodal._ACTIVE_MEDIA_ENV, ""))
        return "decision"

    fake = SimpleNamespace(_benchmark=benchmark)
    multimodal._install_benchmark_policy(fake)
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")

    result = fake._benchmark(
        "llama-server",
        "/tmp/model.gguf",
        _config(),
        object(),
        "fingerprint",
    )

    assert result == "decision"
    assert seen == [""]
    assert os.environ[multimodal._ACTIVE_MEDIA_ENV] == "1"


def test_non_qwen_media_benchmark_keeps_media_scope(monkeypatch) -> None:
    seen: list[str] = []

    def benchmark(_binary, _model_path, _config, _request, _fingerprint):
        seen.append(os.environ.get(multimodal._ACTIVE_MEDIA_ENV, ""))
        return "decision"

    fake = SimpleNamespace(_benchmark=benchmark)
    multimodal._install_benchmark_policy(fake)
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")
    generic = SimpleNamespace(model_id="other/model", extra={"mmproj_filename": "x.gguf"})

    fake._benchmark("llama-server", "/tmp/model.gguf", generic, object(), "fingerprint")

    assert seen == ["1"]


def test_cold_media_final_ensure_keeps_media_scope_and_does_not_prime_twice(
    monkeypatch,
) -> None:
    image = Path("frame.png")
    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect."},),
        media_paths=(image,),
    )
    calls: list[tuple[tuple[Path, ...], str]] = []
    fake = SimpleNamespace(_MANAGED_PROCESS=None, _MANAGED_URL=None)

    def current(_config, current_request):
        calls.append(
            (
                tuple(current_request.media_paths),
                os.environ.get(multimodal._ACTIVE_MEDIA_ENV, ""),
            )
        )
        return "https://external.example/v1"

    def shutdown() -> None:
        raise AssertionError("cold external media must not retire a user-owned server")

    fake.ensure_tuned_server = current
    fake._shutdown_managed_server = shutdown
    multimodal._install_ensure(fake)
    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)

    assert fake.ensure_tuned_server(_config(), request) == "https://external.example/v1"
    assert calls == [((image,), "1")]
    assert os.environ.get(multimodal._ACTIVE_MEDIA_ENV) is None
