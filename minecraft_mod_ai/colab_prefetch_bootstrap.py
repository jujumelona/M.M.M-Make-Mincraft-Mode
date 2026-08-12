from __future__ import annotations

import os
from typing import Any


def start(model_registry_module: Any) -> None:
    """Overlap selected-model prefetch with managed Colab setup only."""
    managed_colab = bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())
    if not managed_colab:
        return

    os.environ.setdefault("MMM_DISCOVERY_WORKERS", "12")
    os.environ.setdefault("MMM_RESEARCH_WORKERS", "8")

    try:
        import __main__

        profile_name = str(getattr(__main__, "MODEL_PROFILE", "")).strip()
    except Exception:
        return
    if not profile_name:
        return

    try:
        registry = model_registry_module.ModelRegistry()
        registry.load_profile(profile_name)
    except Exception:
        # Setup/profile validation remains authoritative in the notebook. Early
        # prefetch is opportunistic and must not replace its explicit diagnostics.
        return


__all__ = ["start"]
