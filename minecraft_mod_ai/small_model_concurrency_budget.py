from __future__ import annotations

"""Bound central model fan-out to the runtime capacity that actually owns the model."""

from contextvars import ContextVar
from functools import wraps
from typing import Any


_MARKER = "_mmm_central_model_capacity_v1"
_ACTIVE_MODEL_ROUTER: ContextVar[Any | None] = ContextVar(
    "mmm_central_model_capacity_router",
    default=None,
)


def harden(agentic_module: Any, central_module: Any) -> None:
    """Reuse central's canonical planner-capacity helper for every model-backed pool.

    Provider retrieval remains governed by the generic CPU/I/O worker budget because
    the router context is installed only around model-backed committee, reviewer, and
    design-section calls.
    """

    current_worker_count = central_module._worker_count
    if not getattr(current_worker_count, _MARKER, False):

        @wraps(current_worker_count)
        def model_aware_worker_count() -> int:
            generic = current_worker_count()
            router = _ACTIVE_MODEL_ROUTER.get()
            if router is None:
                return generic
            try:
                return max(
                    1,
                    min(
                        int(generic),
                        int(central_module._research_domain_worker_count(router, generic)),
                    ),
                )
            except Exception:
                # Unknown model capacity is not permission to oversubscribe one local model.
                return 1

        setattr(model_aware_worker_count, _MARKER, True)
        model_aware_worker_count.__wrapped__ = current_worker_count  # type: ignore[attr-defined]
        central_module._worker_count = model_aware_worker_count

    def wrap_router_scope(owner: Any, name: str) -> None:
        current = getattr(owner, name)
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

    current_generate = agentic_module.generate_sectioned_game_design
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
