from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai.model_adapters.base import GenerationRequest


def _config(*, guarded: bool = True, native_mtp: bool = False) -> SimpleNamespace:
    extra = {"mmproj_filename": "projector.gguf"}
    if guarded:
        extra["runtime_contract"] = "qwen"
        extra["native_mtp"] = native_mtp
    return SimpleNamespace(
        model_id="vendor/arbitrary-runtime-model",
        extra=extra,
    )


def test_cold_media_benchmark_runs_without_media_scope_for_declared_baseline_policy(
    monkeypatch,
) -> None:
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


def test_unconfigured_runtime_keeps_media_scope(monkeypatch) -> None:
    seen: list[str] = []

    def benchmark(_binary, _model_path, _config, _request, _fingerprint):
        seen.append(os.environ.get(multimodal._ACTIVE_MEDIA_ENV, ""))
        return "decision"

    fake = SimpleNamespace(_benchmark=benchmark)
    multimodal._install_benchmark_policy(fake)
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")

    fake._benchmark(
        "llama-server",
        "/tmp/model.gguf",
        _config(guarded=False),
        object(),
        "fingerprint",
    )

    assert seen == ["1"]


def test_native_mtp_runtime_keeps_media_scope(monkeypatch) -> None:
    seen: list[str] = []

    def benchmark(_binary, _model_path, _config, _request, _fingerprint):
        seen.append(os.environ.get(multimodal._ACTIVE_MEDIA_ENV, ""))
        return "decision"

    fake = SimpleNamespace(_benchmark=benchmark)
    multimodal._install_benchmark_policy(fake)
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")

    fake._benchmark(
        "llama-server",
        "/tmp/model.gguf",
        _config(native_mtp=True),
        object(),
        "fingerprint",
    )

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
