from __future__ import annotations

"""CPU-dense retrieval policy enforced at source-owned call and loader boundaries."""

import os
from typing import Any

_DENSE_OPT_IN = "MMM_RAG_ENABLE_CPU_DENSE"


def _dense_opted_in() -> bool:
    return os.environ.get(_DENSE_OPT_IN, "").strip() == "1"


def require_dense_retrieval_device(
    device: Any,
    *,
    role: str,
    model_id: str,
    backend: str,
) -> None:
    """Reject implicit CPU dense retrieval before heavyweight model loading."""

    selected = str(device or "cpu").strip().casefold()
    if not selected.startswith("cpu") or _dense_opted_in():
        return

    from .model_adapters.base import ModelConfigurationError

    raise ModelConfigurationError(
        f"CPU dense retrieval {backend} loading is disabled for role={role!r}, "
        f"model={model_id!r}. Set {_DENSE_OPT_IN}=1 to opt in explicitly."
    )


__all__ = [
    "_dense_opted_in",
    "require_dense_retrieval_device",
]
