from __future__ import annotations

"""Bound central model fan-out to the runtime capacity that actually owns the model."""

from contextvars import ContextVar
from functools import wraps
from typing import Any

_MARKER = "_mmm_central_model_capacity_v3"
_ROUTER_CAPACITY_MARKER = "_mmm_router_owned_model_capacity_v1"
_LEGACY_MARKERS = (
    "_mmm_central_model_capacity_v1",
    "_mmm_central_model_capacity_v2",
)
_ACTIVE_MODEL_ROUTER: ContextVar[Any | None] = ContextVar(
    "mmm_central_model_capacity_router",
    default=None,
)


def _unwrap_legacy(current: Any) -> Any:
    if any(getattr(current, marker, False) for marker in _LEGACY_MARKERS):
        previous = getattr(current, "__wrapped__", None)
        if callable(previous):
            return previous
    return current


def _owns_native_model(router: Any) -> bool:
    """Require router-local proof before process-global llama capacity may be reused."""

    try:
        config = router.registry.role(router.profile, "planner")
    except Exception:
        return False
    return (
        bool(getattr(config, "exclusive_gpu", False))
        and str(getattr(config, "provider", "")) == "local"
        and str(getattr(config, "adapter", "")) in {"llama_cpp", "vllm"}
    )


def harden(agentic_module: Any, central_module: Any) -> None:
    """Reuse central's canonical planner-capacity helper for every model-backed pool.

    Provider retrieval remains governed by the generic CPU/I/O worker budget because
    the router context is installed only around model-backed committee, reviewer, and
    design-section calls. Process-global llama receipts are usable only by a router that
    itself proves ownership of an exclusive native-local planner model.
    """

    current_domain_workers = central_module._research_domain_worker_count
    if not getattr(current_domain_workers, _ROUTER_CAPACITY_MARKER, False):

        @wraps(current_domain_workers)
        def router_owned_domain_workers(router: Any, width: int) -> int:
            try:
                requested = max(1, int(width))
            except (TypeError, ValueError):
                return 1
            if requested <= 1 or not _owns_native_model(router):
                return 1
            try:
                capacity = current_domain_workers(router, requested)
            except Exception:
                return 1
            try:
                return max(1, min(requested, int(capacity)))
            except (TypeError, ValueError):
                return 1

        setattr(router_owned_domain_workers, _ROUTER_CAPACITY_MARKER, True)
        router_owned_domain_workers.__wrapped__ = current_domain_workers  # type: ignore[attr-defined]
        central_module._research_domain_worker_count = router_owned_domain_workers

    current_worker_count = _unwrap_legacy(central_module._worker_count)
    if not getattr(current_worker_count, _MARKER, False):

        @wraps(current_worker_count)
        def model_aware_worker_count() -> int:
            generic = current_worker_count()
            router = _ACTIVE_MODEL_ROUTER.get()
            if router is None:
                return generic

            # _research_domain_worker_count is the existing authority, but it calls
            # central_module._worker_count() internally. Temporarily clear model scope
            # so that nested call sees only the original generic CPU worker budget.
            token = _ACTIVE_MODEL_ROUTER.set(None)
            try:
                capacity = central_module._research_domain_worker_count(router, generic)
            except Exception:
                return 1
            finally:
                _ACTIVE_MODEL_ROUTER.reset(token)
            try:
                return max(1, min(int(generic), int(capacity)))
            except (TypeError, ValueError):
                return 1

        setattr(model_aware_worker_count, _MARKER, True)
        model_aware_worker_count.__wrapped__ = current_worker_count  # type: ignore[attr-defined]
        central_module._worker_count = model_aware_worker_count

    def wrap_router_scope(owner: Any, name: str) -> None:
        current = _unwrap_legacy(getattr(owner, name))
        if getattr(current, _MARKER, False):
            return

        @wraps(current)
        def scoped(router: Any, *args: Any, **kwargs: Any):
            token = _ACTIVE_MODEL_ROUTER.set(router)
            try:
                return current(router, *args, **kwargs)
            finally:
                _ACTIVE_MODEL_ROUTER.reset(token)

        setattr(scoped, _MARKER, True)
        scoped.__wrapped__ = current  # type: ignore[attr-defined]
        setattr(owner, name, scoped)

    wrap_router_scope(central_module, "build_central_committee")
    wrap_router_scope(central_module, "_parallel_reviews")

    current_generate = _unwrap_legacy(agentic_module.generate_sectioned_game_design)
    if not getattr(current_generate, _MARKER, False):

        @wraps(current_generate)
        def generate_scoped(
            game_design_module: Any,
            router: Any,
            prompt: str,
            *,
            media_paths=(),
            research,
            trace_metadata=None,
        ):
            token = _ACTIVE_MODEL_ROUTER.set(router)
            try:
                return current_generate(
                    game_design_module,
                    router,
                    prompt,
                    media_paths=media_paths,
                    research=research,
                    trace_metadata=trace_metadata,
                )
            finally:
                _ACTIVE_MODEL_ROUTER.reset(token)

        setattr(generate_scoped, _MARKER, True)
        generate_scoped.__wrapped__ = current_generate  # type: ignore[attr-defined]
        agentic_module.generate_sectioned_game_design = generate_scoped


__all__ = ["harden"]
