from __future__ import annotations

import os
from typing import Any


def start(model_registry_module: Any) -> None:
    """Apply Colab worker defaults without pretending to prefetch model weights."""
    del model_registry_module
    if not os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        return
    os.environ.setdefault("MMM_DISCOVERY_WORKERS", "12")
    os.environ.setdefault("MMM_RESEARCH_WORKERS", "8")


__all__ = ["start"]
