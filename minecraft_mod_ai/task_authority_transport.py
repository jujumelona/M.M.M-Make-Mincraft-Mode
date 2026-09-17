from __future__ import annotations

"""Low-level validation for host-preserved task-authority transport envelopes."""

from collections.abc import Mapping
from typing import Any

CONTINUATION_REASON = "previous_tool_enabled_page_exhausted_output"


def _continuation_evidence_task(module: Mapping[str, Any]) -> Mapping[str, Any] | None:
    evidence_task = module.get("evidence_task")
    if isinstance(evidence_task, Mapping):
        return evidence_task
    config = module.get("config")
    if not isinstance(config, Mapping):
        return None
    nested = config.get("evidence_task")
    return nested if isinstance(nested, Mapping) else None


def is_preserved_host_continuation(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    continuation = payload.get("continuation")
    module = payload.get("module")
    if not isinstance(continuation, Mapping) or not isinstance(module, Mapping):
        return False
    if str(continuation.get("reason") or "").strip() != CONTINUATION_REASON:
        return False
    index = continuation.get("continuation_index")
    if type(index) is not int or index < 1:
        return False
    module_id = str(module.get("module_id") or "").strip()
    if not module_id:
        return False
    evidence_task = _continuation_evidence_task(module)
    if evidence_task is None:
        return False
    if str(evidence_task.get("task_id") or "").strip() != module_id:
        return False
    bindings = evidence_task.get("production_bindings")
    return isinstance(bindings, list) and bool(bindings)


__all__ = ["CONTINUATION_REASON", "is_preserved_host_continuation"]
