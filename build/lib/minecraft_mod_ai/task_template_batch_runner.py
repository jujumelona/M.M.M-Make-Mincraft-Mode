"""Host grouping for independent record concerns.

Batching here is orchestration only: each concern still gets its own bounded model call.
Small-model prompts remain narrow while independent calls can occupy measured native
llama slots concurrently. Model-side multi-concern output is intentionally not used.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any

from .bounded_record_template import run_bounded_record_template
from .deadline_executor import iter_completed_with_deadlines
from .model_concurrency import router_native_model_parallelism


def supports_record_batching(model_router: Any) -> bool:
    """Use host-side concurrency only when native parallel slots are proven available."""
    return router_native_model_parallelism(model_router) > 1


def record_template_batches(identifiers: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Group one finite concern set for host scheduling, never for one model response."""
    values = tuple(str(identifier) for identifier in identifiers)
    return (values,) if values else ()


def run_record_template_batch(
    model_router: Any,
    identifiers: Sequence[str],
    *,
    context: Mapping[str, Any],
    allowed_refs: set[str],
    progress: Mapping[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, dict[str, Any]]:
    identifiers = tuple(str(identifier) for identifier in identifiers)
    if not identifiers:
        raise ValueError("TEMPLATE_RECORD_BATCH: at least one identifier is required")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("TEMPLATE_RECORD_BATCH: duplicate identifiers")

    checkpoint_lock = RLock()

    def serialized_checkpoint(*args, **kwargs):
        if checkpoint is None:
            return None
        with checkpoint_lock:
            return checkpoint(*args, **kwargs)

    effective_checkpoint = serialized_checkpoint if checkpoint is not None else None

    def run_one(identifier: str) -> dict[str, Any]:
        return run_bounded_record_template(
            model_router,
            identifier,
            context=dict(context),
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=effective_checkpoint,
        )

    workers = max(
        1,
        min(len(identifiers), router_native_model_parallelism(model_router)),
    )
    if workers == 1:
        return {identifier: run_one(identifier) for identifier in identifiers}

    completed: dict[str, dict[str, Any]] = {}
    for identifier, result in iter_completed_with_deadlines(
        identifiers,
        run_one,
        max_workers=workers,
        stage="planning-template-record",
    ):
        completed[identifier] = result

    # Completion order is intentionally decoupled from semantic result order.
    return {identifier: completed[identifier] for identifier in identifiers}


__all__ = [
    "record_template_batches",
    "run_record_template_batch",
    "supports_record_batching",
]
