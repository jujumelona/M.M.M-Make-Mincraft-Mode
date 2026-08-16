from __future__ import annotations

from functools import wraps
from typing import Any


def install(pagination_module: Any) -> None:
    """Reject duplicate production-batch identities instead of renaming them.

    Batch ids participate in dependency edges and exports. Auto-suffixing a duplicate
    changes graph meaning while dependencies still refer to the original id, so the
    planner must repair the invalid page instead of mutating the plan in host code.
    """
    current = pagination_module._append_outline_batches
    if getattr(current, "_mmm_strict_batch_identity", False):
        return

    @wraps(current)
    def append_outline_batches(
        module: Any,
        *,
        raw_batches: list[Any],
        catalog: Any,
        result: list[Any],
    ) -> None:
        for raw in raw_batches:
            if not isinstance(raw, dict):
                raise module.SpecValidationError(
                    "Production outline batch must be a JSON object."
                )
            batch = module._production_batch(raw)
            if batch.batch_id in catalog:
                raise module.SpecValidationError(
                    f"Production outline repeated batch id {batch.batch_id!r}."
                )
            catalog.add(batch.batch_id)
            result.append(batch)

    append_outline_batches._mmm_strict_batch_identity = True  # type: ignore[attr-defined]
    append_outline_batches.__wrapped__ = current  # type: ignore[attr-defined]
    pagination_module._append_outline_batches = append_outline_batches


__all__ = ["install"]
