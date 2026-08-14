from __future__ import annotations

"""Hierarchical workflow/subtask/function procedural memory for temporary skills."""

from collections import Counter
from typing import Any, Mapping, Sequence


def _bucket(record: Mapping[str, Any]) -> tuple[str, str, str]:
    shape = record.get("task_shape")
    shape = shape if isinstance(shape, Mapping) else {}
    stage = str(shape.get("stage", "") or record.get("stage", "") or "general")
    kind = str(shape.get("kind", "") or "general")
    members = shape.get("member_ids")
    function = "general"
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        for item in members:
            value = str(item).strip()
            if value:
                function = value
                break
    return stage, kind, function


def build_hierarchy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    workflow_success: Counter[str] = Counter()
    workflow_fail: Counter[str] = Counter()
    subtask_success: Counter[str] = Counter()
    subtask_fail: Counter[str] = Counter()
    function_success: Counter[str] = Counter()
    function_fail: Counter[str] = Counter()
    for record in records[:32]:
        stage, kind, function = _bucket(record)
        target = record.get("outcome") == "SUCCESS"
        (workflow_success if target else workflow_fail)[stage] += 1
        (subtask_success if target else subtask_fail)[f"{stage}:{kind}"] += 1
        (function_success if target else function_fail)[f"{stage}:{kind}:{function}"] += 1
    return {
        "workflow": {
            "proven": [item for item, _ in workflow_success.most_common(6)],
            "avoid": [item for item, _ in workflow_fail.most_common(6)],
        },
        "subtask": {
            "proven": [item for item, _ in subtask_success.most_common(8)],
            "avoid": [item for item, _ in subtask_fail.most_common(8)],
        },
        "function": {
            "proven": [item for item, _ in function_success.most_common(10)],
            "avoid": [item for item, _ in function_fail.most_common(10)],
        },
    }


def compact_hierarchy(hierarchy: Mapping[str, Any], *, max_items: int = 18) -> dict[str, Any]:
    remaining = max(1, int(max_items))
    result: dict[str, Any] = {}
    for level in ("subtask", "function", "workflow"):
        value = hierarchy.get(level)
        if not isinstance(value, Mapping) or remaining <= 0:
            continue
        proven = list(value.get("proven", ()))[: max(1, remaining // 2)]
        avoid = list(value.get("avoid", ()))[: max(1, remaining // 2)]
        result[level] = {"proven": proven, "avoid": avoid}
        remaining -= len(proven) + len(avoid)
    return result


__all__ = ["build_hierarchy", "compact_hierarchy"]
