from __future__ import annotations

from functools import wraps
from typing import Any


def install() -> None:
    from . import complete_orchestrator_services as services
    from . import model_router as router_module
    from .model_adapters import image_diffusion as image_module

    if not getattr(
        image_module.ImageDiffusionAdapter.generate_image,
        "_mmm_adaptive_image_residency",
        False,
    ):
        raise RuntimeError(
            "ImageDiffusionAdapter must own adaptive pipeline residency natively."
        )
    _install_pipeline_parking(services, router_module, image_module)


def _install_pipeline_parking(
    services: Any,
    router_module: Any,
    image_module: Any,
) -> None:
    """Keep the router's exclusive GPU lease through final image-pipeline parking."""

    original = services.generate_assets
    if getattr(original, "_mmm_adaptive_image_gpu_session", False):
        return

    @wraps(original)
    def adaptive_asset_session(router: Any, *args: Any, **kwargs: Any):
        registry = getattr(router, "registry", None)
        profile = getattr(router, "profile", None)
        if registry is None or profile is None:
            return original(router, *args, **kwargs)
        config = registry.role(profile, "image_generator")
        local_exclusive = (
            config.provider == "local"
            and config.adapter == "image_diffusion"
            and config.exclusive_gpu
        )
        if not local_exclusive:
            return original(router, *args, **kwargs)

        with router_module._GPU_EXCLUSIVE_LOCK:
            try:
                return original(router, *args, **kwargs)
            finally:
                image_module.finish_image_shard()

    adaptive_asset_session._mmm_adaptive_image_gpu_session = True
    services.generate_assets = adaptive_asset_session
