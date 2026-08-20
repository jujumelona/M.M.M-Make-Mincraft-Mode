from __future__ import annotations

from functools import wraps
from typing import Any

from .model_concurrency import (
    ReentrantCapacityGate,
    ReentrantReadWriteLock,
    active_llama_parallelism,
)


_ROUTER_CONTRACT_VERSION = 3
_RESEARCH_DESIGN_CAPACITY_VERSION = 1


def _active_parallelism() -> int:
    """Compatibility name for the canonical model-concurrency policy."""

    return active_llama_parallelism()


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


def install(model_router_module: Any, scheduler_module: Any) -> None:
    """Verify native router concurrency and install remaining scheduler policies."""

    _verify_router(model_router_module)
    _install_scheduler(scheduler_module)
    _install_research_design_capacity_policy(model_router_module)


__all__ = [
    "ReentrantCapacityGate",
    "ReentrantReadWriteLock",
    "_active_parallelism",
    "install",
]
