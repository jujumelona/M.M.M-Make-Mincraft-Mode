from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Iterator, Sequence


def _active_llm_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _dependency_wave_shards(
    work_graph_module: Any,
    modules: Sequence[Any],
    *,
    policy: Any,
) -> Iterator[tuple[str, tuple[Any, ...]]]:
    """Create bounded shards and emit complete dependency-ready waves in order.

    Exact external dependency sets are indexed in O(1). Newly unlocked groups are
    deferred to the next wave so work that was already ready is never overtaken by a
    dependent group. Custom LLM shards expose the selected native decode slots while
    keeping the normal Java/entity shard ceilings.
    """

    staged = [(item, work_graph_module._module_stage(item)) for item in modules]
    stage_counts = Counter(stage for _, stage in staged)
    groups: list[dict[str, Any]] = []
    module_group: dict[str, int] = {}
    open_by_key: dict[tuple[str, frozenset[int]], int] = {}

    def shard_size_for(stage: str) -> int:
        if stage == "entity":
            return max(1, int(policy.entity_shard_size))
        if stage == "custom":
            count = max(1, int(stage_counts[stage]))
            slots = min(_active_llm_slots(), count)
            return min(
                max(1, int(policy.java_shard_size)),
                max(1, math.ceil(count / slots)),
            )
        return max(1, int(policy.java_shard_size))

    for item, stage in staged:
        missing = [
            dependency
            for dependency in item.depends_on
            if dependency not in module_group
        ]
        if missing:
            raise work_graph_module.WorkGraphError(
                "Module sharding requires topological order; unresolved dependencies for "
                f"{item.module_id}: {missing[:4]}"
            )

        shard_size = shard_size_for(stage)
        dependency_groups = {
            module_group[dependency] for dependency in item.depends_on
        }
        candidates: set[int] = set()

        exact_key = (stage, frozenset(dependency_groups))
        exact = open_by_key.get(exact_key)
        if exact is not None and len(groups[exact]["members"]) < shard_size:
            candidates.add(exact)

        for index in dependency_groups:
            group = groups[index]
            if group["stage"] != stage or len(group["members"]) >= shard_size:
                continue
            if (dependency_groups - {index}).issubset(group["external_groups"]):
                candidates.add(index)

        chosen = max(candidates) if candidates else None
        if chosen is None:
            chosen = len(groups)
            external_groups = set(dependency_groups)
            groups.append(
                {
                    "stage": stage,
                    "members": [],
                    "external_groups": external_groups,
                    "first_order": len(module_group),
                }
            )
            open_by_key[(stage, frozenset(external_groups))] = chosen

        group = groups[chosen]
        group["members"].append(item)
        module_group[item.module_id] = chosen

        group_key = (str(group["stage"]), frozenset(group["external_groups"]))
        if len(group["members"]) >= shard_size:
            if open_by_key.get(group_key) == chosen:
                open_by_key.pop(group_key, None)
        else:
            previous = open_by_key.get(group_key)
            if previous is None or chosen > previous:
                open_by_key[group_key] = chosen

    dependents: dict[int, list[int]] = {index: [] for index in range(len(groups))}
    indegree = [0] * len(groups)
    for index, group in enumerate(groups):
        dependencies = sorted(set(group["external_groups"]))
        indegree[index] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(index)

    ready = sorted(
        (index for index, degree in enumerate(indegree) if degree == 0),
        key=lambda index: int(groups[index]["first_order"]),
    )
    emitted = 0
    while ready:
        next_ready: set[int] = set()
        for index in ready:
            group = groups[index]
            yield str(group["stage"]), tuple(group["members"])
            emitted += 1
            for dependent in dependents[index]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.add(dependent)
        ready = sorted(
            next_ready,
            key=lambda index: int(groups[index]["first_order"]),
        )

    if emitted != len(groups):
        raise work_graph_module.WorkGraphError(
            "Module shard dependency graph contains a cycle."
        )


def install(*, work_graph_module: Any) -> None:
    """Install dependency-aware work-graph sharding exactly once."""

    current_shards = work_graph_module._module_shards
    if getattr(current_shards, "_mmm_dependency_wave_shards", False):
        return

    def module_shards(modules: Sequence[Any], *, policy: Any):
        yield from _dependency_wave_shards(
            work_graph_module,
            modules,
            policy=policy,
        )

    module_shards._mmm_dependency_wave_shards = True  # type: ignore[attr-defined]
    work_graph_module._module_shards = module_shards


__all__ = ["install", "_active_llm_slots", "_dependency_wave_shards"]
