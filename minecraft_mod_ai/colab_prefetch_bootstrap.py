from __future__ import annotations

import os
from typing import Any


def start(model_registry_module: Any) -> None:
    """Resolve the selected Colab profile early so its GGUF prefetch can overlap I/O."""

    if not os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        return

    # A seed page carries at most twelve independent ecosystem routes. Start all
    # twelve by default in Colab instead of leaving four queued behind an 8-worker
    # pool. Explicit user/runtime overrides remain authoritative.
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
