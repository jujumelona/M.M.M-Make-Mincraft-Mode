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


def _adaptive_expand_one_production_batch_factory(module: Any):
    def adaptive_expand_one_production_batch(
        self: Any,
        *,
        batch: Any,
        parts: Any,
        module_catalog: Any,
        asset_catalog: Any,
        audio_catalog: Any,
        test_catalog: set[str],
        dependency_exports: dict[str, list[str]],
        planning_context: dict[str, Any],
        planning_receipt: dict[str, Any],
        media_paths: Sequence[Any],
    ) -> None:
        """Use adaptive page width and durable object-level semantic repair."""

        from .production_page_durable_contract import (
            load_or_generate_page,
            resolve_page_items,
        )

        remaining = list(batch.deliverables)
        cursor = ""
        first_page = True
        seen_states: set[tuple[tuple[str, ...], str]] = set()

        while remaining:
            state = (tuple(remaining), cursor)
            if state in seen_states:
                raise module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} pagination made no progress."
                )
            seen_states.add(state)

            target_deliverables = list(remaining)
            request = {
                "batch": {
                    "batch_id": batch.batch_id,
                    "scope": batch.scope,
                    "depends_on_batches": list(batch.depends_on_batches),
                    "deliverables": list(batch.deliverables),
                    "exports": list(batch.exports),
                },
                "current_target_deliverable": target_deliverables[0],
                "current_target_deliverables": target_deliverables,
                "remaining_deliverables": list(remaining),
                "total_remaining": len(remaining),
                "dependency_exports": dependency_exports,
                "planning_context_receipt": planning_receipt,
                "known_module_catalog": module_catalog.receipt(),
                "known_asset_catalog": asset_catalog.receipt(),
                "known_audio_catalog": audio_catalog.receipt(),
                "cursor": cursor,
                "contract": module._PRODUCTION_PAGE_CONTRACT,
            }
            if first_page:
                request["planning_context"] = planning_context

            stage = f"production batch {batch.batch_id!r} page"

            def generate_page() -> dict[str, Any]:
                return module._generate_json_page_with_repair(
                    self.router,
                    system_prompt=(
                        "Return one clean production-batch JSON page. The host provides ALL "
                        "currently outstanding deliverables in current_target_deliverables. "
                        "There is no fixed host item count. Choose a coherent NON-EMPTY subset "
                        "that you can fully implement before the output limit; if all remaining "
                        "work fits comfortably, complete all of it. Prefer another continuation "
                        "page over truncating or padding the response. Every claimed completed "
                        "deliverable must be backed by emitted module/asset/audio/test evidence; "
                        "use implements_deliverables on emitted objects when applicable. Never "
                        "repeat IDs already committed by the host catalogs."
                    ),
                    request=request,
                    media_paths=media_paths if first_page else (),
                    expected_contracts=(frozenset(module._PRODUCTION_PAGE_CONTRACT),),
                    stage=stage,
                )

            page, page_path = load_or_generate_page(
                stage=stage,
                request=request,
                generate=generate_page,
            )
            first_page = False

            if set(page) != set(module._PRODUCTION_PAGE_CONTRACT):
                raise module.SpecValidationError(
                    "Production batch page fields are invalid."
                )

            page_modules, page_assets, page_audio, tests = resolve_page_items(
                module,
                self.router,
                page=page,
                page_path=page_path,
                module_catalog=module_catalog,
                asset_catalog=asset_catalog,
                audio_catalog=audio_catalog,
                test_catalog=test_catalog,
            )

            parts.modules.extend(page_modules)
            parts.assets.extend(page_assets)
            parts.audio.extend(page_audio)
            parts.acceptance_tests.extend(tests)
            test_catalog.update(tests)

            completed_raw = page.get("completed_deliverables", [])
            remaining_set = set(remaining)
            completed = {
                str(value).strip()
                for value in completed_raw
                if isinstance(value, str) and str(value).strip() in remaining_set
            }
            if not completed:
                raise module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} page made no host-verifiable "
                    "deliverable progress."
                )

            remaining = [value for value in remaining if value not in completed]
            cursor_value = page.get("next_cursor", "")
            cursor = cursor_value if isinstance(cursor_value, str) else ""

    adaptive_expand_one_production_batch._mmm_adaptive_page_width = True  # type: ignore[attr-defined]
    adaptive_expand_one_production_batch._mmm_durable_production_items = True  # type: ignore[attr-defined]
    return adaptive_expand_one_production_batch


def _dependency_wave_shards(
    work_graph_module: Any,
    modules: Sequence[Any],
    *,
    policy: Any,
) -> Iterator[tuple[str, tuple[Any, ...]]]:
    """Create bounded shards and emit complete dependency-ready waves in order.

    Consecutive modules may share one durable shard only when doing so cannot add an
    unrelated external dependency. Exact external dependency sets are indexed in O(1),
    while chain extension checks only groups that the current module actually depends
    on. Emission is level-synchronous: groups already ready at the start of a wave are
    all emitted before any group unlocked by that wave.

    Custom LLM work is sized to expose the native decode slots selected by autotuning,
    while the normal Java/entity shard ceilings keep SQLite/checkpoint overhead bounded.
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


def install(*, complete_planner_module: Any, work_graph_module: Any) -> None:
    """Remove proven serial critical-path waste without weakening dependency fences."""

    planner_cls = complete_planner_module.CompleteGameDesignPlanner
    current_expand = planner_cls._expand_one_production_batch
    if not getattr(current_expand, "_mmm_durable_production_items", False):
        planner_cls._expand_one_production_batch = (
            _adaptive_expand_one_production_batch_factory(complete_planner_module)
        )

    current_shards = work_graph_module._module_shards
    if not getattr(current_shards, "_mmm_dependency_wave_shards", False):

        def module_shards(modules: Sequence[Any], *, policy: Any):
            yield from _dependency_wave_shards(
                work_graph_module,
                modules,
                policy=policy,
            )

        module_shards._mmm_dependency_wave_shards = True  # type: ignore[attr-defined]
        work_graph_module._module_shards = module_shards


__all__ = ["install", "_active_llm_slots", "_dependency_wave_shards"]
