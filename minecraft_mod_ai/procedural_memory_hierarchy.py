from __future__ import annotations

"""Workflow -> subtask -> function procedural memory from verified action traces."""

from collections import Counter
from typing import Any, Mapping, Sequence

from .procedure_trace import sequence_actions


def _bucket(record: Mapping[str, Any]) -> tuple[str, str, str]:
    shape = record.get("task_shape")
    shape = shape if isinstance(shape, Mapping) else {}
    workflow = str(shape.get("stage", "") or record.get("stage", "") or "general")
    subtask = str(shape.get("kind", "") or "general")
    members = shape.get("member_ids")
    function = "general"
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        for item in members:
            value = str(item).strip()
            if value:
                function = value
                break
    return workflow, subtask, function


def _sequence_windows(actions: Sequence[str], width: int) -> list[str]:
    if not actions:
        return []
    if len(actions) <= width:
        return [" > ".join(actions)]
    return [" > ".join(actions[index : index + width]) for index in range(len(actions) - width + 1)]


def build_hierarchy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate verified ordered procedures at three reusable abstraction levels."""

    workflow_success: Counter[str] = Counter()
    workflow_fail: Counter[str] = Counter()
    subtask_success: Counter[str] = Counter()
    subtask_fail: Counter[str] = Counter()
    function_success: Counter[str] = Counter()
    function_fail: Counter[str] = Counter()
    workflow_sequences_success: Counter[str] = Counter()
    workflow_sequences_fail: Counter[str] = Counter()
    subtask_sequences_success: Counter[str] = Counter()
    subtask_sequences_fail: Counter[str] = Counter()
    function_steps_success: Counter[str] = Counter()
    function_steps_fail: Counter[str] = Counter()

    for record in records[:32]:
        workflow, subtask, function = _bucket(record)
        success = record.get("outcome") == "SUCCESS"
        actions = sequence_actions(
            record.get("procedure") if isinstance(record.get("procedure"), Mapping) else None
        )

        (workflow_success if success else workflow_fail)[workflow] += 1
        (subtask_success if success else subtask_fail)[f"{workflow}:{subtask}"] += 1
        (function_success if success else function_fail)[f"{workflow}:{subtask}:{function}"] += 1

        # Workflow memory keeps the complete bounded action chain; subtask memory
        # keeps local 2-3 step motifs; function memory keeps concrete transition
        # edges.  This is procedural sequence memory, not label-only taxonomy.
        if actions:
            full = " > ".join(actions[:12])
            (workflow_sequences_success if success else workflow_sequences_fail)[
                f"{workflow} :: {full}"
            ] += 1
            for motif in _sequence_windows(actions[:16], 3):
                (subtask_sequences_success if success else subtask_sequences_fail)[
                    f"{workflow}:{subtask} :: {motif}"
                ] += 1
            for action in actions[:24]:
                (function_steps_success if success else function_steps_fail)[
                    f"{workflow}:{subtask}:{function} :: {action}"
                ] += 1

    return {
        "workflow": {
            "proven": [item for item, _ in workflow_success.most_common(6)],
            "avoid": [item for item, _ in workflow_fail.most_common(6)],
            "procedures": [item for item, _ in workflow_sequences_success.most_common(6)],
            "failed_procedures": [item for item, _ in workflow_sequences_fail.most_common(6)],
        },
        "subtask": {
            "proven": [item for item, _ in subtask_success.most_common(8)],
            "avoid": [item for item, _ in subtask_fail.most_common(8)],
            "procedures": [item for item, _ in subtask_sequences_success.most_common(8)],
            "failed_procedures": [item for item, _ in subtask_sequences_fail.most_common(8)],
        },
        "function": {
            "proven": [item for item, _ in function_success.most_common(10)],
            "avoid": [item for item, _ in function_fail.most_common(10)],
            "procedures": [item for item, _ in function_steps_success.most_common(10)],
            "failed_procedures": [item for item, _ in function_steps_fail.most_common(10)],
        },
    }


def compact_hierarchy(hierarchy: Mapping[str, Any], *, max_items: int = 18) -> dict[str, Any]:
    remaining = max(1, int(max_items))
    result: dict[str, Any] = {}
    # Concrete function/subtask motifs are most useful to a small model; workflow
    # chains remain as compact global guidance after those are selected.
    for level in ("function", "subtask", "workflow"):
        value = hierarchy.get(level)
        if not isinstance(value, Mapping) or remaining <= 0:
            continue
        level_result: dict[str, list[str]] = {}
        for key in ("procedures", "failed_procedures", "proven", "avoid"):
            if remaining <= 0:
                break
            raw = value.get(key, ())
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                continue
            take = min(len(raw), max(1, remaining // 3))
            selected = [str(item) for item in raw[:take] if str(item).strip()]
            if selected:
                level_result[key] = selected
                remaining -= len(selected)
        if level_result:
            result[level] = level_result
    return result


__all__ = ["build_hierarchy", "compact_hierarchy"]
