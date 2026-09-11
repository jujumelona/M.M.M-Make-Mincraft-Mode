"""Host grouping for independent record concerns.

Batching here is orchestration only: each concern still gets its own bounded model call.
This keeps small-model prompts narrow and removes model-owned batch/continuation control.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .bounded_record_template import run_bounded_record_template


def supports_record_batching(model_router: Any) -> bool:
    """Transport-level multi-concern batching is intentionally disabled."""
    return False


def record_template_batches(identifiers: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Return one host-owned concern per model call, with no arbitrary batch width."""
    return tuple((str(identifier),) for identifier in identifiers)


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

    return {
        identifier: run_bounded_record_template(
            model_router,
            identifier,
            context=dict(context),
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=checkpoint,
        )
        for identifier in identifiers
    }


__all__ = [
    "record_template_batches",
    "run_record_template_batch",
    "supports_record_batching",
]
