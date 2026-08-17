from __future__ import annotations

"""VRAM-first admission for the managed llama-server.

The native runtime validates concurrent slot variants with live requests and falls
back after launch failures. This policy only refines runtime resource admission;
it does not own Planner or repair search width.
"""

import json
import os
from functools import wraps
from typing import Any, Callable

_POLICY_VERSION = 3
_MIB = 1024 * 1024
_RESOURCE_MARKER = "_mmm_vram_parallel_resource_policy_v3"
_SELECTION_MARKER = "_mmm_vram_parallel_selection_policy_v3"
_LEGACY_RESOURCE_MARKERS = (
    "_mmm_vram_parallel_resource_policy_v1",
    "_mmm_vram_parallel_resource_policy_v2",
)
_LEGACY_SELECTION_MARKERS = (
    "_mmm_vram_parallel_selection_policy_v1",
    "_mmm_vram_parallel_selection_policy_v2",
)
_RUNTIME_RECEIPT_SCHEMA = "mmm/llama-runtime-receipt-v1"


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


def validated_active_parallelism() -> int:
    """Return slots only when the managed runtime receipt proves they are live."""
    active = _active_parallelism()
    if active <= 1:
        return 1
    raw = os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "").strip()
    if not raw:
        return 1
    try:
        receipt = json.loads(raw)
    except Exception:
        return 1
    if not isinstance(receipt, dict):
        return 1
    if str(receipt.get("schema_version", "")) != _RUNTIME_RECEIPT_SCHEMA:
        return 1
    try:
        receipt_slots = max(1, min(8, int(receipt.get("slots", 1))))
    except (TypeError, ValueError):
        return 1
    return active if receipt_slots == active else 1


def _unwrap_marked(current: Callable[..., Any], markers: tuple[str, ...]) -> Callable[..., Any]:
    while any(bool(getattr(current, marker, False)) for marker in markers):
        previous = getattr(current, "__wrapped__", None)
        if not callable(previous):
            break
        current = previous
    return current


def _install_resource_admission(runtime_tuning: Any) -> None:
    current = runtime_tuning._parallel_resource_feasible
    if getattr(current, _RESOURCE_MARKER, False):
        return
    current = _unwrap_marked(current, _LEGACY_RESOURCE_MARKERS)

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
    current = _unwrap_marked(current, _LEGACY_SELECTION_MARKERS)

    @wraps(current)
    def selection_inputs(config: Any) -> dict[str, Any]:
        payload = dict(current(config))
        payload["vram_parallel_policy_version"] = _POLICY_VERSION
        return payload

    setattr(selection_inputs, _SELECTION_MARKER, True)
    runtime_tuning._selection_inputs = selection_inputs


def install(runtime_tuning: Any) -> None:
    """Install only the VRAM-first runtime policy, idempotently."""
    _install_resource_admission(runtime_tuning)
    _install_selection_version(runtime_tuning)


__all__ = ["install", "validated_active_parallelism"]
