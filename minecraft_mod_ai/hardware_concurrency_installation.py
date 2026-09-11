from __future__ import annotations

import os
from typing import Any

_MAX_AUTO_CPU_IO_WORKERS = 32
_MAX_EXPLICIT_CPU_IO_WORKERS = 128
_MARKER = "_mmm_hardware_concurrency_v1"


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def recommended_cpu_io_workers(*, cpu_count: int | None = None) -> int:
    """Scale CPU/I/O fan-out to the host while leaving one logical CPU for the model/OS."""

    explicit = _positive_int(os.environ.get("MMM_CPU_IO_WORKERS"))
    if explicit is not None:
        return max(1, min(_MAX_EXPLICIT_CPU_IO_WORKERS, explicit))

    logical = _positive_int(cpu_count)
    if logical is None:
        logical = _positive_int(os.cpu_count()) or 2
    if logical <= 2:
        return logical
    return max(2, min(_MAX_AUTO_CPU_IO_WORKERS, logical - 1))


def recommended_central_ai_workers() -> int:
    """Expose the full reviewed model fan-out; runtime VRAM receipts remain authoritative."""

    explicit = _positive_int(os.environ.get("MMM_CENTRAL_AI_WORKERS"))
    if explicit is not None:
        return max(1, min(8, explicit))
    active = _positive_int(os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL")) or 8
    return max(1, min(8, active))


def install() -> dict[str, int]:
    """Install host-adaptive orchestration widths without bypassing model safety gates."""

    central_workers = recommended_central_ai_workers()
    os.environ.setdefault("MMM_CENTRAL_AI_WORKERS", str(central_workers))

    cpu_workers = recommended_cpu_io_workers()
    os.environ["MMM_CPU_IO_WORKERS_EFFECTIVE"] = str(cpu_workers)

    try:
        from . import scheduler_parallel_safety_contract as scheduler
    except Exception:
        return {
            "cpu_io_workers": cpu_workers,
            "central_ai_workers": central_workers,
        }

    current = scheduler._cpu_capacity
    if not getattr(current, _MARKER, False):

        def adaptive_cpu_capacity() -> int:
            return recommended_cpu_io_workers()

        setattr(adaptive_cpu_capacity, _MARKER, True)
        adaptive_cpu_capacity.__wrapped__ = current  # type: ignore[attr-defined]
        scheduler._cpu_capacity = adaptive_cpu_capacity

    return {
        "cpu_io_workers": cpu_workers,
        "central_ai_workers": central_workers,
    }
