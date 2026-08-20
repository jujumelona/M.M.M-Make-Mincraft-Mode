from __future__ import annotations

"""Compatibility verifier for the native image residency contract."""


def install() -> None:
    from . import complete_orchestrator_services as services
    from . import model_router
    from .model_adapters import image_diffusion as image_module

    if not getattr(
        image_module.ImageDiffusionAdapter.generate_image,
        "_mmm_adaptive_image_residency",
        False,
    ):
        raise RuntimeError(
            "ImageDiffusionAdapter must own adaptive pipeline residency natively."
        )
    if not hasattr(model_router.ModelRouter, "image_generation_session"):
        raise RuntimeError("ModelRouter must own the image GPU session natively.")
    if not getattr(services.generate_assets, "_mmm_adaptive_image_gpu_session", False):
        raise RuntimeError("Asset generation must enter the native image GPU session.")


__all__ = ["install"]
