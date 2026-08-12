from __future__ import annotations

import threading
from types import SimpleNamespace

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

    services = SimpleNamespace(generate_assets=generate_assets)
    model_router = SimpleNamespace(_GPU_EXCLUSIVE_LOCK=lock)
    install(services_module=services, model_router_module=model_router)

    assert services.generate_assets(router) == "ok"
    assert observed == [True]
