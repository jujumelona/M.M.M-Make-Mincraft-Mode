from __future__ import annotations

"""Durable work-node boundary handoff for an exhausted llama action.

Context pressure is owned by ``llama_length_resilience``.  The producer that owns the
temporary module workspace owns the other typed ``finish_reason=length`` case and may
continue from edits already staged there.  The durable-node boundary must never replay
the whole action: doing so discards that workspace and repeats the same completion and
its retrieval/tool side effects from the beginning.

This compatibility installer deliberately preserves the typed exception chain when a
same-workspace producer cannot make bounded progress.  That lets the normal durable
failure path report the real boundary without manufacturing a second model request.
"""

from functools import wraps
from typing import Any

_MARKER = "_mmm_completion_boundary_work_handoff_v2"
_LEGACY_REPLAY_MARKER = "_mmm_completion_boundary_work_recovery_v1"


def install(orchestrator_cls: type[Any]) -> None:
    """Install a single-call durable boundary without whole-action replay."""

    current = orchestrator_cls._run_work_node
    if getattr(current, _MARKER, False):
        return

    # A notebook may re-run runtime finalization after upgrading the checkout in the
    # same Python process.  Peel the former replay wrapper before installing v2;
    # otherwise the obsolete hidden retry would remain inside this no-replay layer.
    if getattr(current, _LEGACY_REPLAY_MARKER, False):
        current = getattr(current, "__wrapped__", current)

    @wraps(current)
    def run_with_boundary_handoff(
        ledger: Any,
        node: Any,
        *,
        action: Any,
        validate_cached: Any,
        shared_index: Any = None,
    ) -> dict[str, Any]:
        return current(
            ledger,
            node,
            action=action,
            validate_cached=validate_cached,
            shared_index=shared_index,
        )

    setattr(run_with_boundary_handoff, _MARKER, True)
    orchestrator_cls._run_work_node = staticmethod(run_with_boundary_handoff)


__all__ = ["install"]
