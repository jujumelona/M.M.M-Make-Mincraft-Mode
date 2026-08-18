from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal
from minecraft_mod_ai.model_adapters.base import GenerationRequest


class _RunningProcess:
    @staticmethod
    def poll():
        return None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        extra={
            "gguf_filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
        },
    )


def test_cold_media_primes_text_autotune_before_media_scope(monkeypatch) -> None:
    image = Path("frame.png")
    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect."},),
        media_paths=(image,),
    )
    events: list[tuple[tuple[Path, ...], str]] = []
    shutdowns: list[str] = []
    fake = SimpleNamespace(_MANAGED_PROCESS=None, _MANAGED_URL=None)

    def current(_config, current_request):
        active = os.environ.get(multimodal._ACTIVE_MEDIA_ENV, "")
        events.append((tuple(current_request.media_paths), active))
        fake._MANAGED_PROCESS = _RunningProcess()
        port = 8920 if current_request.media_paths else 8910
        fake._MANAGED_URL = f"http://127.0.0.1:{port}/v1"
        os.environ["LLAMA_SERVER_URL"] = fake._MANAGED_URL
        return fake._MANAGED_URL

    def shutdown() -> None:
        shutdowns.append(str(fake._MANAGED_URL))
        fake._MANAGED_PROCESS = None
        fake._MANAGED_URL = None

    fake.ensure_tuned_server = current
    fake._shutdown_managed_server = shutdown
    multimodal._install_ensure(fake)
    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)

    result = fake.ensure_tuned_server(_config(), request)

    assert events == [
        ((), ""),
        ((image,), "1"),
    ]
    assert shutdowns == ["http://127.0.0.1:8910/v1"]
    assert result == "http://127.0.0.1:8920/v1"
    assert os.environ.get(multimodal._ACTIVE_MEDIA_ENV) is None
    assert getattr(fake, multimodal._MANAGED_MEDIA_PROCESS_ATTR) is fake._MANAGED_PROCESS


def test_cold_media_does_not_kill_user_owned_external_server(monkeypatch) -> None:
    image = Path("frame.png")
    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect."},),
        media_paths=(image,),
    )
    calls: list[tuple[tuple[Path, ...], str]] = []
    shutdowns: list[bool] = []
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
        shutdowns.append(True)

    fake.ensure_tuned_server = current
    fake._shutdown_managed_server = shutdown
    multimodal._install_ensure(fake)
    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)

    assert fake.ensure_tuned_server(_config(), request) == "https://external.example/v1"
    assert calls == [
        ((), ""),
        ((image,), "1"),
    ]
    assert shutdowns == []
