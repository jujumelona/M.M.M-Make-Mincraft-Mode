from __future__ import annotations

import math
import os
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
        """Use adaptive page width and durable object-level semantic repair.

        Splitting a coherent production batch into arbitrary host-sized request pages
        cannot create useful decode parallelism; it only adds prompt/decode overhead.
        Present every outstanding deliverable, let the model finish any coherent
        non-empty subset that fits cleanly, persist the returned page before semantic
        parsing, and patch only individual invalid item fields.
        """

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
    """Shard only modules that are simultaneously dependency-ready.

    Consecutive topological order is not a readiness wave: placing a dependent module
    in the same coarse shard as an unrelated ready module makes unrelated work wait for
    the dependency. Compute dependency depth first, then shard within (depth, stage).

    Custom LLM work is split just enough to occupy the native decode slots selected by
    autotuning, not one durable row per module. The normal Java shard ceiling still
    bounds very large waves, so SQLite/checkpoint overhead remains sublinear in project
    size while small waves no longer collapse all custom decoding into one serial node.
    """

    levels: dict[str, int] = {}
    buckets: dict[tuple[int, str], list[Any]] = {}
    stage_order: dict[int, list[str]] = {}

    for item in modules:
        missing = [dependency for dependency in item.depends_on if dependency not in levels]
        if missing:
            raise work_graph_module.WorkGraphError(
                "Module sharding requires topological order; unresolved dependencies for "
                f"{item.module_id}: {missing[:4]}"
            )
        level = (
            0
            if not item.depends_on
            else 1 + max(levels[dependency] for dependency in item.depends_on)
        )
        levels[item.module_id] = level
        stage = work_graph_module._module_stage(item)
        key = (level, stage)
        buckets.setdefault(key, []).append(item)
        stages = stage_order.setdefault(level, [])
        if stage not in stages:
            stages.append(stage)

    for level in sorted(stage_order):
        for stage in stage_order[level]:
            values = buckets[(level, stage)]
            if stage == "entity":
                shard_size = policy.entity_shard_size
            elif stage == "custom":
                slots = min(_active_llm_slots(), len(values))
                shard_size = min(
                    policy.java_shard_size,
                    max(1, math.ceil(len(values) / slots)),
                )
            else:
                shard_size = policy.java_shard_size
            for index in range(0, len(values), shard_size):
                yield stage, tuple(values[index : index + shard_size])


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
