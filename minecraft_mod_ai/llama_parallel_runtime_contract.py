from __future__ import annotations

import json
import os
from functools import wraps
from typing import Any

from .model_concurrency import (
    ReentrantCapacityGate,
    ReentrantReadWriteLock,
    active_llama_parallelism,
)

_ROUTER_CONTRACT_VERSION = 3
_RESEARCH_DESIGN_CAPACITY_VERSION = 1
_DYNAMIC_RUNTIME_PARALLEL_VERSION = 1
_MIB = 1024 * 1024


def _active_parallelism() -> int:
    """Compatibility name for the canonical model-concurrency policy."""

    return active_llama_parallelism()


def _positive_env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _verify_router(model_router_module: Any) -> None:
    """Fail closed if an old router is loaded instead of patching it at runtime."""

    version = int(
        getattr(
            model_router_module.ModelRouter.generate_text,
            "_mmm_parallel_router_contract_version",
            0,
        )
        or 0
    )
    if version < _ROUTER_CONTRACT_VERSION:
        raise RuntimeError(
            "ModelRouter must own native llama concurrency directly; "
            "runtime method replacement is no longer supported."
        )
    if not isinstance(model_router_module._GPU_EXCLUSIVE_LOCK, ReentrantReadWriteLock):
        raise RuntimeError("ModelRouter GPU lock does not support shared native inference.")
    if not isinstance(model_router_module._LLAMA_INFERENCE_SLOTS, ReentrantCapacityGate):
        raise RuntimeError("ModelRouter llama inference capacity gate is missing.")


def _install_scheduler(scheduler_module: Any) -> None:
    current = scheduler_module._capacities
    if getattr(current, "_mmm_dynamic_llama_slots", False):
        return

    @wraps(current)
    def capacities() -> dict[str, int]:
        values = dict(current())
        values["llm"] = _active_parallelism()
        return values

    capacities._mmm_dynamic_llama_slots = True  # type: ignore[attr-defined]
    scheduler_module._capacities = capacities


def _install_research_design_capacity_policy(model_router_module: Any) -> None:
    """Apply managed-runtime limits only to routers that own the llama process."""

    from . import central_intelligence_amplifier as central_module

    current = central_module._research_domain_worker_count
    installed_version = int(
        getattr(current, "_mmm_managed_research_capacity_version", 0) or 0
    )
    if installed_version >= _RESEARCH_DESIGN_CAPACITY_VERSION:
        return

    @wraps(current)
    def research_design_capacity(router: Any, width: int) -> int:
        requested = min(max(1, int(width)), central_module._worker_count())
        if isinstance(router, model_router_module.ModelRouter):
            return current(router, width)
        try:
            config = router.registry.role(router.profile, "planner")
        except Exception:
            return requested
        if not bool(getattr(config, "exclusive_gpu", False)):
            return 1
        if str(getattr(config, "provider", "")) != "local":
            return 1
        if str(getattr(config, "adapter", "")) not in {"llama_cpp", "vllm"}:
            return 1
        return requested

    research_design_capacity._mmm_managed_research_capacity_version = (  # type: ignore[attr-defined]
        _RESEARCH_DESIGN_CAPACITY_VERSION
    )
    research_design_capacity.__wrapped__ = current  # type: ignore[attr-defined]
    central_module._research_domain_worker_count = research_design_capacity


def _install_dynamic_runtime_parallelism(runtime_tuning: Any, vram_policy: Any) -> None:
    """Replace the legacy fixed slot ceiling with resource-derived admission.

    The managed server has a single authoritative safety boundary: total context,
    live free VRAM, live available RAM, and (when supplied) the requested concurrent
    workload.  No second numeric slot cap is allowed in the host scheduler.
    """

    current_candidates = getattr(runtime_tuning, "_parallel_candidates", None)
    installed = int(
        getattr(current_candidates, "_mmm_dynamic_parallel_version", 0) or 0
    )
    if installed >= _DYNAMIC_RUNTIME_PARALLEL_VERSION:
        return

    def total_context(per_request: int, slots: int) -> int:
        slots = max(1, int(slots))
        per_request = max(0, int(per_request))
        if per_request <= 0 and slots > 1:
            raise RuntimeError(
                "parallel llama-server requires a positive per-request context so slots do not share an unknown context"
            )
        total = per_request * slots
        maximum = max(1, int(getattr(runtime_tuning, "_MAX_TOTAL_CONTEXT", 0) or 0))
        if total > maximum:
            raise RuntimeError(
                f"llama-server total context {total} exceeds supported maximum {maximum}"
            )
        return total

    def explicit_parallel() -> int | None:
        return _positive_env_int("MMM_LLAMA_PARALLEL")

    def parallel_target() -> int:
        requested = _positive_env_int("MMM_LLAMA_CONCURRENT_REQUESTS")
        if requested is not None:
            return requested
        return 1 if runtime_tuning._performance_mode() == "latency" else 1

    def resource_ceiling(
        config: Any,
        model_path: str | None,
        resources: Any | None = None,
    ) -> int:
        snapshot = resources or runtime_tuning._runtime_resources()
        try:
            per_request = int(runtime_tuning._per_request_context(config))
            model_bytes = int(runtime_tuning._model_size(model_path))
            gpu_free = max(0, int(snapshot.gpu_free_bytes))
            ram_available = max(0, int(snapshot.ram_available_bytes))
            kv_bytes = max(1, int(runtime_tuning._kv_bytes_per_token()))
            maximum_context = max(
                1, int(getattr(runtime_tuning, "_MAX_TOTAL_CONTEXT", 0) or 0)
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return 1

        if per_request <= 0 or model_bytes <= 0 or gpu_free <= 0 or ram_available <= 0:
            return 1

        context_bound = maximum_context // per_request
        gpu_budget = int(gpu_free * 0.97) - int(model_bytes * 1.07) - 1280 * _MIB
        gpu_per_slot = per_request * kv_bytes
        gpu_bound = gpu_budget // gpu_per_slot if gpu_budget > 0 else 0
        ram_budget = int(ram_available * 0.94) - 512 * _MIB
        ram_bound = ram_budget // (256 * _MIB) if ram_budget > 0 else 0
        ceiling = max(1, min(context_bound, gpu_bound, ram_bound))

        requested = _positive_env_int("MMM_LLAMA_CONCURRENT_REQUESTS")
        if requested is not None:
            ceiling = min(ceiling, requested)
        return max(1, int(ceiling))

    def parallel_resource_feasible(
        slots: int,
        config: Any,
        model_path: str | None,
        resources: Any,
    ) -> bool:
        slots = max(1, int(slots))
        if slots == 1:
            return True
        if os.environ.get("MMM_LLAMA_VRAM_PARALLEL", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return slots <= resource_ceiling(config, model_path, resources)

    def parallel_candidates(
        config: Any | None = None,
        model_path: str | None = None,
        resources: Any | None = None,
    ) -> tuple[int, ...]:
        explicit = explicit_parallel()
        if explicit is not None:
            return (explicit,)
        if runtime_tuning._performance_mode() == "latency" and _positive_env_int(
            "MMM_LLAMA_CONCURRENT_REQUESTS"
        ) is None:
            return (1,)
        if config is None:
            return (1,)

        maximum = resource_ceiling(config, model_path, resources)
        values = [1]
        value = 2
        while value < maximum:
            values.append(value)
            value *= 2
        if maximum > 1 and values[-1] != maximum:
            values.append(maximum)
        return tuple(values)

    def recommended_parallel(
        tuning: Any,
        config: Any,
        model_path: str,
    ) -> int:
        explicit = explicit_parallel()
        if explicit is not None:
            return explicit
        if tuning._performance_mode() == "latency" and _positive_env_int(
            "MMM_LLAMA_CONCURRENT_REQUESTS"
        ) is None:
            return 1
        raw_maximize = os.environ.get("MMM_LLAMA_MAXIMIZE_PARALLEL", "1").strip().lower()
        if raw_maximize in {"0", "false", "no", "off"}:
            return 1
        return resource_ceiling(config, model_path, tuning._runtime_resources())

    def policy_active_parallelism() -> int:
        return active_llama_parallelism()

    def validated_active_parallelism() -> int:
        active = policy_active_parallelism()
        if active <= 1:
            return 1
        raw = os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "").strip()
        if not raw:
            return 1
        try:
            receipt = json.loads(raw)
            receipt_slots = max(1, int(receipt.get("slots", 1)))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return 1
        if not isinstance(receipt, dict):
            return 1
        if str(receipt.get("schema_version", "")) != str(
            getattr(vram_policy, "_RUNTIME_RECEIPT_SCHEMA", "")
        ):
            return 1
        return active if receipt_slots == active else 1

    total_context._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]
    explicit_parallel._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]
    parallel_target._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]
    parallel_resource_feasible._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]
    parallel_candidates._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]
    recommended_parallel._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]
    validated_active_parallelism._mmm_dynamic_parallel_version = _DYNAMIC_RUNTIME_PARALLEL_VERSION  # type: ignore[attr-defined]

    runtime_tuning._total_context = total_context
    runtime_tuning._explicit_parallel = explicit_parallel
    runtime_tuning._parallel_target = parallel_target
    runtime_tuning._parallel_resource_feasible = parallel_resource_feasible
    runtime_tuning._parallel_candidates = parallel_candidates
    vram_policy._active_parallelism = policy_active_parallelism
    vram_policy.validated_active_parallelism = validated_active_parallelism
    vram_policy._recommended_parallel = recommended_parallel


def install(model_router_module: Any, scheduler_module: Any) -> None:
    """Verify native router concurrency and install remaining scheduler policies."""

    from . import llama_server_runtime_tuning, llama_vram_parallel_policy

    _verify_router(model_router_module)
    _install_dynamic_runtime_parallelism(
        llama_server_runtime_tuning,
        llama_vram_parallel_policy,
    )
    _install_scheduler(scheduler_module)
    _install_research_design_capacity_policy(model_router_module)


__all__ = [
    "ReentrantCapacityGate",
    "ReentrantReadWriteLock",
    "_active_parallelism",
    "_install_dynamic_runtime_parallelism",
    "install",
]
