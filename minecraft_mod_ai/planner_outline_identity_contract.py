from __future__ import annotations

from functools import wraps
from typing import Any


def install(pagination_module: Any) -> None:
    """Do not silently rename duplicate production-batch identities.

    Batch ids participate in dependency edges and exports. Renaming a duplicate with
    ``_2`` changes graph meaning while dependencies still refer to the original id.
    Duplicate identity is therefore invalid planner output and must be repaired, not
    normalized into a different plan.
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
