from __future__ import annotations

"""Colab-specific asynchronous warmup and worker-budget defaults."""

import os
from pathlib import Path
from typing import Any


def start(model_registry_module: Any) -> None:
    """Start native metadata warmup; never replace runtime implementations."""
    del model_registry_module
    from .platform_live_discovery import start_platform_prefetch

    start_platform_prefetch()
    if not os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        return
    discovery_workers, research_workers = _colab_worker_defaults()
    os.environ.setdefault("MMM_DISCOVERY_WORKERS", str(discovery_workers))
    os.environ.setdefault("MMM_RESEARCH_WORKERS", str(research_workers))


def _colab_worker_defaults() -> tuple[int, int]:
    """Size network and CPU retrieval pools from the live Colab allocation."""
    cpu_count = max(1, int(os.cpu_count() or 1))
    available_gib = 4
    try:
        for raw_line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if not raw_line.startswith("MemAvailable:"):
                continue
            available_kib = int(raw_line.split()[1])
            available_gib = max(1, available_kib // (1024 * 1024))
            break
    except (OSError, ValueError, IndexError):
        pass
    discovery = min(12, max(4, cpu_count * 4), max(4, available_gib * 2))
    research = min(8, max(2, cpu_count * 2), max(2, available_gib))
    return max(1, discovery), max(1, research)


__all__ = ["_colab_worker_defaults", "start"]
