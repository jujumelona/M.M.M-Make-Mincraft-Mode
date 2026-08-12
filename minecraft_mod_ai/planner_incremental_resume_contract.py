from __future__ import annotations

from functools import wraps
from typing import Any


def install(incremental_module: Any) -> None:
    """Resume saved queues, patch duplicates, and finalize without regeneration."""

    # Install critical-path efficiency after planner JSON/repair wrappers exist. Keep
    # the local GPU at one decode slot, remove host-imposed planner page width and
    # dependency-head-of-line sharding, journal large checkpoints in linear time,
    # checkpoint individual image/audio sources inside coarse residency-friendly
    # shards, batch large-DAG scheduler polling, losslessly salvage and resume
    # truncated production streams, and escalate best-of-N LLM search only after a
    # verifier state actually persists.
    from . import agentic_optimization_contract as agentic_module
    from . import audio_generator as audio_module
    from . import complete_orchestrator as orchestrator_module
    from . import complete_orchestrator_services as orchestrator_services
    from . import complete_planner as complete_planner_module
    from . import planner_json_runtime_contract as planner_runtime_module
    from . import work_graph as work_graph_module
    from .agentic_search_efficiency_contract import install as install_agentic_search_efficiency
    from .asset_resume_efficiency_contract import install as install_asset_resume_efficiency
    from .audio_resume_efficiency_contract import install as install_audio_resume_efficiency
    from .execution_efficiency_contract import install as install_execution_efficiency
    from .planner_checkpoint_journal_contract import install as install_checkpoint_journal
    from .production_stream_efficiency_contract import (
        install as install_production_stream_efficiency,
    )
    from .production_stream_resume_contract import (
        install as install_production_stream_resume,
    )
    from .scheduler_poll_efficiency_contract import install as install_scheduler_poll_efficiency

    install_checkpoint_journal(incremental_module)
    install_agentic_search_efficiency(agentic_module)
    install_asset_resume_efficiency(orchestrator_services)
    install_audio_resume_efficiency(audio_module)
    install_scheduler_poll_efficiency(work_graph_module)

    # complete_orchestrator imports synthesize_audio_files directly, before this late
    # planner/architecture bootstrap installs the resumable audio wrapper. Rebind that
    # already-imported hot-path name so audio-synth DAG nodes actually use the per-sound
    # checkpoint instead of merely leaving the optimized function on audio_generator.
    orchestrator_module.synthesize_audio_files = audio_module.synthesize_audio_files

    # Disable the historical production 2->1 host width reduction at its source too.
    # The lossless stream wrapper below already bypasses that loop, but making this
    # helper an identity prevents any retained/internal callable from reintroducing a
    # fixed target count. Page width belongs to the model; the host owns validation,
    # persistence, salvage and continuation only.
    def no_production_width_narrowing(request: Any, attempt: int) -> Any:
        del attempt
        return request

    no_production_width_narrowing._mmm_no_fixed_production_width = True  # type: ignore[attr-defined]
    planner_runtime_module._narrow_production_repair_request = no_production_width_narrowing

    # These wrappers are intentionally installed after planner_json_runtime_contract.
    # Production uses lossless stream salvage, then an outer disk-resume layer so a
    # crash after fsync but before salvage never causes another full GPU decode. All
    # non-production structured planner pages retain their existing behavior.
    install_production_stream_efficiency(complete_planner_module)
    install_production_stream_resume(complete_planner_module)
    install_execution_efficiency(
        complete_planner_module=complete_planner_module,
        work_graph_module=work_graph_module,
    )

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
