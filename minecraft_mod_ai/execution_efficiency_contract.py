from __future__ import annotations

import heapq
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
            completed = {
                str(value).strip()
                for value in completed_raw
                if isinstance(value, str) and str(value).strip() in set(remaining)
            }
            if not completed:
                raise module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} page made no verified progress."
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
    """Create bounded shards without exploding long serial dependency chains.

    Dependencies between members of one shard are safe because ``modules`` is already
    topological and each bounded member list is processed deterministically in order.
    The unsafe case is adding an otherwise-ready module to a shard that has unrelated
    external dependencies. To avoid that artificial wait, a module may join an open
    shard only when its effective external dependency-group set is identical.

    This keeps ready work separate while compressing a 20k linear chain into bounded
    ``java_shard_size`` nodes instead of 20k SQLite rows. Custom LLM work additionally
    sizes independent shards to expose the native decode slots selected by autotuning.
    """

    staged = [(item, work_graph_module._module_stage(item)) for item in modules]
    stage_counts = Counter(stage for _, stage in staged)
    groups: list[dict[str, Any]] = []
    module_group: dict[str, int] = {}

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
        missing = [dependency for dependency in item.depends_on if dependency not in module_group]
        if missing:
            raise work_graph_module.WorkGraphError(
                "Module sharding requires topological order; unresolved dependencies for "
                f"{item.module_id}: {missing[:4]}"
            )

        shard_size = shard_size_for(stage)
        chosen: int | None = None
        for index in range(len(groups) - 1, -1, -1):
            group = groups[index]
            if group["stage"] != stage or len(group["members"]) >= shard_size:
                continue

            internal_dependency = any(
                module_group[dependency] == index
                for dependency in item.depends_on
            )
            effective_external = {
                module_group[dependency]
                for dependency in item.depends_on
                if module_group[dependency] != index
            }
            if internal_dependency:
                effective_external.update(group["external_groups"])

            if effective_external == group["external_groups"]:
                chosen = index
                break

        if chosen is None:
            chosen = len(groups)
            groups.append(
                {
                    "stage": stage,
                    "members": [],
                    "external_groups": {
                        module_group[dependency] for dependency in item.depends_on
                    },
                    "first_order": len(module_group),
                }
            )

        groups[chosen]["members"].append(item)
        module_group[item.module_id] = chosen

    dependents: dict[int, list[int]] = {index: [] for index in range(len(groups))}
    indegree = [0] * len(groups)
    for index, group in enumerate(groups):
        dependencies = sorted(set(group["external_groups"]))
        indegree[index] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(index)

    ready: list[tuple[int, int]] = []
    for index, degree in enumerate(indegree):
        if degree == 0:
            heapq.heappush(ready, (int(groups[index]["first_order"]), index))

    emitted = 0
    while ready:
        _, index = heapq.heappop(ready)
        group = groups[index]
        yield str(group["stage"]), tuple(group["members"])
        emitted += 1
        for dependent in dependents[index]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(
                    ready,
                    (int(groups[dependent]["first_order"]), dependent),
                )

    if emitted != len(groups):
        raise work_graph_module.WorkGraphError("Module shard dependency graph contains a cycle.")


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
