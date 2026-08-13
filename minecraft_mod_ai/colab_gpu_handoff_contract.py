from __future__ import annotations

import os
from functools import wraps
from typing import Any


def _release_native_llama_server() -> None:
    """Release the managed native text server before another exclusive GPU runtime."""

    from . import llama_server_autotune

    process = getattr(llama_server_autotune, "_MANAGED_PROCESS", None)
    if process is None or process.poll() is not None:
        return
    managed_url = getattr(llama_server_autotune, "_MANAGED_URL", None)
    llama_server_autotune._shutdown_managed_server()
    if managed_url and os.environ.get("LLAMA_SERVER_URL") == managed_url:
        os.environ.pop("LLAMA_SERVER_URL", None)
    llama_server_autotune._ATTEMPTED_KEYS.clear()


def _install_asset_handoff(*, services_module: Any, model_router_module: Any) -> None:
    current = services_module.generate_assets
    if getattr(current, "_mmm_atomic_gpu_handoff", False):
        return

    @wraps(current)
    def generate_assets_atomic_gpu_handoff(router: Any, *args: Any, **kwargs: Any):
        registry = getattr(router, "registry", None)
        profile = getattr(router, "profile", None)
        local_exclusive_image = False
        if registry is not None and profile is not None:
            try:
                config = registry.role(profile, "image_generator")
                local_exclusive_image = (
                    config.provider == "local"
                    and config.adapter == "image_diffusion"
                    and config.exclusive_gpu
                )
            except Exception:
                local_exclusive_image = False
        if not local_exclusive_image:
            return current(router, *args, **kwargs)
        # Own the same GPU lock used by text generation, then evict the managed
        # llama-server exactly once before the asset shard starts. Image residency
        # keeps the diffusion pipeline alive for all images/tiles in this shard, so
        # neither model is repeatedly loaded and evicted per image.
        with model_router_module._GPU_EXCLUSIVE_LOCK:
            _release_native_llama_server()
            return current(router, *args, **kwargs)

    generate_assets_atomic_gpu_handoff._mmm_atomic_gpu_handoff = True  # type: ignore[attr-defined]
    services_module.generate_assets = generate_assets_atomic_gpu_handoff


def _install_speech_handoff(*, model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    current = cls.transcribe
    if getattr(current, "_mmm_atomic_speech_gpu_handoff", False):
        return

    @wraps(current)
    def transcribe_atomic_gpu_handoff(self: Any, role: str, audio_path: Any) -> str:
        config = self.registry.role(self.profile, role)
        local_exclusive_speech = (
            config.provider == "local"
            and config.adapter == "speech"
            and config.exclusive_gpu
        )
        if not local_exclusive_speech:
            return current(self, role, audio_path)
        with model_router_module._GPU_EXCLUSIVE_LOCK:
            _release_native_llama_server()
            return current(self, role, audio_path)

    transcribe_atomic_gpu_handoff._mmm_atomic_speech_gpu_handoff = True  # type: ignore[attr-defined]
    cls.transcribe = transcribe_atomic_gpu_handoff


def install(*, services_module: Any, model_router_module: Any) -> None:
    """Serialize native-server eviction with local exclusive GPU consumers."""

    _install_asset_handoff(
        services_module=services_module,
        model_router_module=model_router_module,
    )
    _install_speech_handoff(model_router_module=model_router_module)


__all__ = ["install"]
