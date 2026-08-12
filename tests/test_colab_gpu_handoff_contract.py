from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.colab_gpu_handoff_contract import install


def test_local_asset_wrapper_holds_global_gpu_lock_during_handoff() -> None:
    lock = threading.RLock()
    observed: list[bool] = []

    class Registry:
        def role(self, profile, role):
            assert role == "image_generator"
            return SimpleNamespace(
                provider="local",
                adapter="image_diffusion",
                exclusive_gpu=True,
            )

    router = SimpleNamespace(registry=Registry(), profile="test")

    def generate_assets(router, *args, **kwargs):
        is_owned = getattr(lock, "_is_owned", None)
        assert callable(is_owned)
        observed.append(bool(is_owned()))
        return "ok"

    class ModelRouter:
        def transcribe(self, role, audio_path):
            return "transcript"

    services = SimpleNamespace(generate_assets=generate_assets)
    model_router = SimpleNamespace(_GPU_EXCLUSIVE_LOCK=lock, ModelRouter=ModelRouter)
    install(services_module=services, model_router_module=model_router)

    assert services.generate_assets(router) == "ok"
    assert observed == [True]


def test_local_speech_evicts_resident_llama_under_same_gpu_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.RLock()
    events: list[tuple[str, bool]] = []

    class Registry:
        def role(self, profile, role):
            assert role == "speech_recognition"
            return SimpleNamespace(
                provider="local",
                adapter="speech",
                exclusive_gpu=True,
            )

    class ModelRouter:
        def __init__(self) -> None:
            self.registry = Registry()
            self.profile = "test"

        def transcribe(self, role, audio_path):
            is_owned = getattr(lock, "_is_owned", None)
            assert callable(is_owned)
            events.append(("transcribe", bool(is_owned())))
            return "transcript"

    services = SimpleNamespace(generate_assets=lambda *args, **kwargs: None)
    model_router = SimpleNamespace(_GPU_EXCLUSIVE_LOCK=lock, ModelRouter=ModelRouter)

    from minecraft_mod_ai import colab_mtp_server

    monkeypatch.setattr(colab_mtp_server, "colab_mtp_server_enabled", lambda: True)

    def fake_stop(*, keep_enabled: bool = True):
        is_owned = getattr(lock, "_is_owned", None)
        assert callable(is_owned)
        events.append(("stop", bool(is_owned())))

    monkeypatch.setattr(colab_mtp_server, "stop_colab_mtp_server", fake_stop)
    install(services_module=services, model_router_module=model_router)

    assert ModelRouter().transcribe("speech_recognition", "/tmp/a.wav") == "transcript"
    assert events == [("stop", True), ("transcribe", True)]
