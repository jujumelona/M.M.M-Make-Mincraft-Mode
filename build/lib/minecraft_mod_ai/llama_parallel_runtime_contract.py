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
_DYNAMIC_RUNTIME_PARALLEL_VERSION = 2


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


def _install_dynamic_runtime_parallelism(runtime_tuning: Any, vram_policy: Any) -> None:
    """Verify one canonical runtime/VRAM parallelism implementation.

    Older revisions replaced ``runtime_tuning`` and ``vram_policy`` functions here at
    import time.  That created two independent slot policies: explicit widths could skip
    the runtime owner's validation/clamping, and the VRAM fast-start selector was replaced
    by a second implementation with different semantics.  The runtime tuning module now
    owns slot parsing/resource feasibility and the VRAM policy owns fast-start admission.
    This integration layer therefore validates those owners instead of monkey-patching
    them.
    """

    required_runtime = (
        "_total_context",
        "_explicit_parallel",
        "_parallel_target",
        "_parallel_resource_feasible",
        "_parallel_candidates",
    )
    missing = [name for name in required_runtime if not callable(getattr(runtime_tuning, name, None))]
    if missing:
        raise RuntimeError(
            "llama runtime tuning contract is incomplete: " + ", ".join(missing)
        )
    if not callable(getattr(vram_policy, "_recommended_parallel", None)):
        raise RuntimeError("llama VRAM policy does not own _recommended_parallel")
    if not callable(getattr(vram_policy, "validated_active_parallelism", None)):
        raise RuntimeError("llama VRAM policy does not expose validated active slots")

    # Marker only: no executable function is replaced here.  Keeping the marker on the
    # integration function makes idempotence/diagnostics observable without changing the
    # canonical owners' identities.
    _install_dynamic_runtime_parallelism._mmm_dynamic_parallel_version = (  # type: ignore[attr-defined]
        _DYNAMIC_RUNTIME_PARALLEL_VERSION
    )


def install(model_router_module: Any, scheduler_module: Any) -> None:
    """Verify native router concurrency and install scheduler-facing policies."""

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
