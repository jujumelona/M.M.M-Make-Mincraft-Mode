from __future__ import annotations

"""VRAM-first admission and plan fan-out for the managed llama-server.

The native runtime already validates p1/p2/p4 variants with real concurrent
requests and falls back after launch failures. This policy only removes a
duplicated host-RAM model reserve from *automatic extra-slot admission* and
feeds every successfully activated llama slot during planner candidate search.
"""

import os
from functools import wraps
from typing import Any

_POLICY_VERSION = 1
_MIB = 1024 * 1024
_RESOURCE_MARKER = "_mmm_vram_parallel_resource_policy_v1"
_SELECTION_MARKER = "_mmm_vram_parallel_selection_policy_v1"
_PLANNER_MARKER = "_mmm_fill_active_llama_slots_v1"


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _active_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _install_resource_admission(runtime_tuning: Any) -> None:
    current = runtime_tuning._parallel_resource_feasible
    if getattr(current, _RESOURCE_MARKER, False):
        return

    @wraps(current)
    def vram_first_parallel_feasible(
        slots: int,
        config: Any,
        model_path: str | None,
        resources: Any,
    ) -> bool:
        if current(slots, config, model_path, resources):
            return True

        slots = max(1, int(slots))
        if slots <= 1 or not _env_enabled("MMM_LLAMA_VRAM_PARALLEL", True):
            return False

        # Extra llama-server slots share the one mmap/offloaded model. The old
        # gate charged 40% of model size again against MemAvailable for every
        # p2/p4 admission decision. On a T4 Colab this can reject all extra
        # slots while several GiB of VRAM are idle. Keep the full GPU/KV model
        # budget and a conservative server/slot host-RAM reserve; the existing
        # live parallel probe and sequential p4->p2->p1 launch fallback remain
        # the final safety authority.
        try:
            context = runtime_tuning._per_request_context(config)
            total_context = runtime_tuning._total_context(context, slots)
            model_bytes = runtime_tuning._model_size(model_path)
            gpu_free = max(0, int(resources.gpu_free_bytes))
            ram_available = max(0, int(resources.ram_available_bytes))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

        if not model_bytes or not gpu_free or not ram_available:
            return False

        gpu_required = (
            int(model_bytes * 1.07)
            + total_context * runtime_tuning._kv_bytes_per_token()
            + 1280 * _MIB
        )
        host_runtime_required = (512 + 256 * slots) * _MIB
        return bool(
            gpu_required <= int(gpu_free * 0.92)
            and host_runtime_required <= int(ram_available * 0.90)
        )

    setattr(vram_first_parallel_feasible, _RESOURCE_MARKER, True)
    runtime_tuning._parallel_resource_feasible = vram_first_parallel_feasible


def _install_selection_version(runtime_tuning: Any) -> None:
    current = runtime_tuning._selection_inputs
    if getattr(current, _SELECTION_MARKER, False):
        return

    @wraps(current)
    def selection_inputs(config: Any) -> dict[str, Any]:
        payload = dict(current(config))
        payload["vram_parallel_policy_version"] = _POLICY_VERSION
        return payload

    setattr(selection_inputs, _SELECTION_MARKER, True)
    runtime_tuning._selection_inputs = selection_inputs


def _install_planner_slot_filling(runtime_tuning: Any, agentic_module: Any) -> None:
    current = agentic_module._planner_candidate_count
    if getattr(current, _PLANNER_MARKER, False):
        return

    @wraps(current)
    def planner_candidate_count(request: Any, stage: str) -> int:
        width = max(1, int(current(request, stage)))
        if not _env_enabled("MMM_PLAN_FILL_ACTIVE_LLAMA_SLOTS", True):
            return width
        if os.environ.get("MMM_PLAN_SEARCH_WIDTH", "").strip():
            return width
        try:
            if agentic_module._mode() == "off":
                return width
        except Exception:
            pass
        try:
            if runtime_tuning._performance_mode() == "latency":
                return width
        except Exception:
            pass

        # MMM_LLAMA_ACTIVE_PARALLEL is exported only after the managed server
        # has actually launched a validated slot count. Never fan out beyond
        # that live capacity.
        return max(width, _active_parallelism())

    setattr(planner_candidate_count, _PLANNER_MARKER, True)
    agentic_module._planner_candidate_count = planner_candidate_count


def install(runtime_tuning: Any, agentic_module: Any) -> None:
    """Install the VRAM-first policy idempotently."""
    _install_resource_admission(runtime_tuning)
    _install_selection_version(runtime_tuning)
    _install_planner_slot_filling(runtime_tuning, agentic_module)


__all__ = ["install"]
