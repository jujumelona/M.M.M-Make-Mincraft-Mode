from __future__ import annotations

import os
from functools import wraps
from typing import Any


def install(game_design_module: Any) -> None:
    """Prevent the legacy notebook's internal 1.20.1 placeholder from pinning plans.

    platform_api_contract must still pass a historical value into the old constructor
    for ABI compatibility. The generated Colab notebook also used to pass 1.20.1
    explicitly. In a managed Colab run that value is an implementation placeholder,
    not a user constraint. A user-written version in the natural-language request is
    still visible to the normal platform resolver and remains authoritative.
    """

    cls = game_design_module.GameDesignPlanner
    current = cls.plan
    if getattr(current, "_mmm_colab_auto_platform_placeholder", False):
        return

    @wraps(current)
    def plan(self: Any, prompt: str, *, media_paths=()):
        managed_colab = bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())
        router = getattr(self, "router", None)
        placeholder = (
            managed_colab
            and router is not None
            and str(
                getattr(router, "_mmm_requested_minecraft_version", "") or ""
            ).strip()
            == "1.20.1"
            and str(
                getattr(router, "_mmm_requested_loader", "fabric") or "fabric"
            ).strip().casefold()
            == "fabric"
        )
        if not placeholder:
            return current(self, prompt, media_paths=media_paths)

        previous_version = getattr(router, "_mmm_requested_minecraft_version", None)
        previous_loader = getattr(router, "_mmm_requested_loader", None)
        try:
            if hasattr(router, "_mmm_requested_minecraft_version"):
                delattr(router, "_mmm_requested_minecraft_version")
            if hasattr(router, "_mmm_requested_loader"):
                delattr(router, "_mmm_requested_loader")
            return current(self, prompt, media_paths=media_paths)
        finally:
            router._mmm_requested_minecraft_version = previous_version
            router._mmm_requested_loader = previous_loader

    plan._mmm_colab_auto_platform_placeholder = True
    cls.plan = plan


__all__ = ["install"]
