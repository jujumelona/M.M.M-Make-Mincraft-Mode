from __future__ import annotations

import os
from typing import Any

from .scheduler_parallel_safety_contract import recommended_cpu_io_workers


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def recommended_central_ai_workers() -> int:
    """Expose model fan-out only when runtime parallelism has been validated."""

    explicit = _positive_int(os.environ.get("MMM_CENTRAL_AI_WORKERS"))
    if explicit is not None:
        return explicit
    active = _positive_int(os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL"))
    return active or 1


def install() -> dict[str, int]:
    """Publish effective widths without freezing future validated runtime capacity."""

    central_workers = recommended_central_ai_workers()
    # MMM_CENTRAL_AI_WORKERS is an operator override.  Do not populate it with the
    # pre-launch fallback (usually 1), otherwise later validated llama parallelism
    # can never become visible to the central-AI scheduler.
    os.environ["MMM_CENTRAL_AI_WORKERS_EFFECTIVE"] = str(central_workers)

    cpu_workers = recommended_cpu_io_workers()
    os.environ["MMM_CPU_IO_WORKERS_EFFECTIVE"] = str(cpu_workers)

    return {
        "cpu_io_workers": cpu_workers,
        "central_ai_workers": central_workers,
    }
