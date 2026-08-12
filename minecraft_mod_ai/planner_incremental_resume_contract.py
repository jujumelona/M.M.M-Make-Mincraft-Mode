from __future__ import annotations

from functools import wraps
from typing import Any


def install(incremental_module: Any) -> None:
    """Resume saved queues, patch duplicates, and finalize without regeneration."""

    current = incremental_module._process_pending_batches
    if getattr(current, "_mmm_contextual_pending_queue", False):
        return

    @wraps(current)
    def process_with_context(*args: Any, **kwargs: Any) -> None:
        runtime_module = args[0] if len(args) > 0 else kwargs.get("runtime_module")
        module = args[1] if len(args) > 1 else kwargs.get("module")
        router = args[2] if len(args) > 2 else kwargs.get("router")
        saved_batches = kwargs.get("saved_batches")
        checkpoint_path = kwargs.get("checkpoint_path")
        checkpoint_state = kwargs.get("checkpoint_state")

        if (
            runtime_module is None
            or module is None
            or router is None
            or not isinstance(saved_batches, list)
            or checkpoint_path is None
            or not isinstance(checkpoint_state, dict)
        ):
            current(*args, **kwargs)
            return

        pending = list(checkpoint_state.get("pending_batches", []))
        while pending:
            candidate: Any = pending[0]
            while True:
                accepted_ids = incremental_module._accepted_batch_ids(saved_batches)
                error = incremental_module._batch_validation_error(module, candidate)
                candidate_id = (
                    str(candidate.get("batch_id", "")).strip()
                    if isinstance(candidate, dict)
                    else ""
                )
                if not error and candidate_id and candidate_id in accepted_ids:
                    error = (
                        f"duplicate batch_id {candidate_id!r}; this id is already saved. "
                        "Change only batch_id to a new descriptive snake_case id."
                    )

                if not error and isinstance(candidate, dict):
                    resolved = dict(candidate)
                    break

                resolved = incremental_module._patch_one_invalid_batch(
                    runtime_module,
                    module,
                    router,
                    raw_batch=candidate,
                    validation_error=error or "batch must be a JSON object",
                    accepted_batch_ids=accepted_ids,
                    checkpoint_path=checkpoint_path,
                    checkpoint_state=checkpoint_state,
                )
                # Revalidate contextual constraints too. If the patch kept a duplicate
                # id, patch that field again rather than silently dropping the batch.
                candidate = resolved

            incremental_module._merge_saved_batches(saved_batches, [resolved])
            pending.pop(0)
            checkpoint_state.update(
                {
                    "saved_batches": saved_batches,
                    "pending_batches": pending,
                    "pending_patch": None,
                    "status": "collecting",
                }
            )
            incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)

        checkpoint_state["saved_batches"] = saved_batches
        checkpoint_state["pending_patch"] = None
        checkpoint_state["status"] = (
            "complete" if checkpoint_state.get("page_complete") else "page_complete"
        )
        incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)

    process_with_context._mmm_contextual_pending_queue = True  # type: ignore[attr-defined]
    incremental_module._process_pending_batches = process_with_context


__all__ = ["install"]
