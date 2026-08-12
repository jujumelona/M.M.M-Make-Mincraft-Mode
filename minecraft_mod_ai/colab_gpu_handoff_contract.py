from __future__ import annotations

from functools import wraps
from typing import Any


def install(*, services_module: Any, model_router_module: Any) -> None:
    """Make server eviction and local asset generation one exclusive GPU action.

    The existing asset wrapper correctly evicts the resident llama process before
    loading FLUX, but it did so before ModelRouter.generate_image acquired the global
    GPU lock. A concurrently running LLM could therefore be killed from another
    executor. This outer re-entrant lock covers both eviction and the nested image
    generation call while CPU/I/O lanes remain fully concurrent.
    """

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
        # which enters the same global GPU scope again on this thread.
        with model_router_module._GPU_EXCLUSIVE_LOCK:
            return current(router, *args, **kwargs)

    generate_assets_atomic_gpu_handoff._mmm_atomic_gpu_handoff = True
    services_module.generate_assets = generate_assets_atomic_gpu_handoff


__all__ = ["install"]
