from __future__ import annotations

from functools import wraps
from typing import Any


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

        # RLock is intentional: current() eventually calls router.generate_image(),
        # which enters the same global GPU scope again on this thread. The inner
        # asset wrapper evicts the resident llama server while this outer lock is
        # held, so a concurrent text request cannot be killed mid-decode.
        with model_router_module._GPU_EXCLUSIVE_LOCK:
            return current(router, *args, **kwargs)

    generate_assets_atomic_gpu_handoff._mmm_atomic_gpu_handoff = True
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

        # Whisper is instantiated on CUDA inside SpeechAdapter. A resident llama
        # process can hold most VRAM even while idle, so lock + eviction must happen
        # before the speech pipeline is constructed. The text server remains enabled
        # logically and is recreated by the next local text request.
        with model_router_module._GPU_EXCLUSIVE_LOCK:
            try:
                from .colab_mtp_server import (
                    colab_mtp_server_enabled,
                    stop_colab_mtp_server,
                )

                if colab_mtp_server_enabled():
                    stop_colab_mtp_server(keep_enabled=True)
            except Exception:
                # The underlying speech call remains authoritative. On a non-Colab
                # host there may be no managed llama process to evict.
                pass
            return current(self, role, audio_path)

    transcribe_atomic_gpu_handoff._mmm_atomic_speech_gpu_handoff = True
    cls.transcribe = transcribe_atomic_gpu_handoff


def install(*, services_module: Any, model_router_module: Any) -> None:
    """Serialize resident-model eviction with local exclusive GPU consumers."""

    _install_asset_handoff(
        services_module=services_module,
        model_router_module=model_router_module,
    )
    _install_speech_handoff(model_router_module=model_router_module)


__all__ = ["install"]
