from __future__ import annotations

"""Durable work-node recovery for a genuinely exhausted llama action.

Context pressure is owned by ``llama_length_resilience``.  This layer handles the
other typed ``finish_reason=length`` case: an assistant action that consumed its full
output allowance before completing.  Retrying the durable work node recreates the
module staging workspace and lets the normal bounded source-edit ACI split work across
fresh agent turns; it does not issue a hidden second completion inside the adapter.
"""

from functools import wraps
from typing import Any

from .llama_finish_reason_contract import OUTPUT_EXHAUSTED, completion_boundary_kind

_MARKER = "_mmm_completion_boundary_work_recovery_v1"


def install(orchestrator_cls: type[Any]) -> None:
    """Retry exactly one durable work-node action on typed output exhaustion."""

    current = orchestrator_cls._run_work_node
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def run_with_boundary_recovery(
        ledger: Any,
        node: Any,
        *,
        action: Any,
        validate_cached: Any,
        shared_index: Any = None,
    ) -> dict[str, Any]:
        try:
            return current(
                ledger,
                node,
                action=action,
                validate_cached=validate_cached,
                shared_index=shared_index,
            )
        except BaseException as exc:
            if completion_boundary_kind(exc) != OUTPUT_EXHAUSTED:
                raise

            task = ledger.task(node.node_id)
            if str(task.get("state", "")) == "failed":
                ledger.retry(node.node_id)

            return current(
                ledger,
                node,
                action=action,
                validate_cached=validate_cached,
                shared_index=shared_index,
            )

    setattr(run_with_boundary_recovery, _MARKER, True)
    orchestrator_cls._run_work_node = staticmethod(run_with_boundary_recovery)


__all__ = ["install"]
