from __future__ import annotations

from typing import Any, Iterator, Sequence


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
        """Let the planner choose useful page width instead of forcing groups of four.

        The local llama server has one decode slot, so splitting a coherent batch into
        host-sized groups cannot create parallel speedup; it only adds serial prompt/
        decode overhead.  Present every outstanding deliverable, let the model finish
        any coherent non-empty subset that fits cleanly, and let host-verified evidence
        decide what remains for the next page.
        """

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

            # No arbitrary host width. The model sees all outstanding work and chooses
            # how much can be completed without truncating the JSON object.
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

            page = module._generate_json_page_with_repair(
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
                stage=f"production batch {batch.batch_id!r} page",
            )
            first_page = False

            if set(page) != set(module._PRODUCTION_PAGE_CONTRACT):
                raise module.SpecValidationError(
                    "Production batch page fields are invalid."
                )

            raw_modules = module._list(page, "modules")
            raw_assets = module._list(page, "assets")
            raw_audio = module._list(page, "audio")
            raw_tests = module._list(page, "acceptance_tests")

            page_modules = [
                module._module(item) for item in raw_modules if isinstance(item, dict)
            ]
            page_assets = [
                module._asset(item) for item in raw_assets if isinstance(item, dict)
            ]
            page_audio = [
                module._audio(item) for item in raw_audio if isinstance(item, dict)
            ]
            tests = [
                str(value).strip()
                for value in raw_tests
                if str(value).strip() and str(value).strip() not in test_catalog
            ]

            for value in page_modules:
                module_catalog.add(value.module_id)
            for value in page_assets:
                asset_catalog.add(value.asset_id)
            for value in page_audio:
                audio_catalog.add(value.sound_id)

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
                # The runtime normally rejects this before returning the page. Keep the
                # invariant local as well so future wrappers cannot create an infinite
                # continuation loop.
                raise module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} page made no verified progress."
                )

            remaining = [value for value in remaining if value not in completed]
            cursor_value = page.get("next_cursor", "")
            cursor = cursor_value if isinstance(cursor_value, str) else ""

    adaptive_expand_one_production_batch._mmm_adaptive_page_width = True  # type: ignore[attr-defined]
    return adaptive_expand_one_production_batch


def _dependency_wave_shards(
    work_graph_module: Any,
    modules: Sequence[Any],
    *,
    policy: Any,
) -> Iterator[tuple[str, tuple[Any, ...]]]:
    """Shard only modules that are simultaneously dependency-ready.

    Consecutive topological order is not a readiness wave: placing a dependent module
    in the same coarse shard as an unrelated ready module makes the unrelated work wait
    for the dependency.  Compute dependency depth first, then shard within (depth,stage).
    Custom LLM modules use one durable node each; the LLM lane is already capacity one,
    so this improves checkpoint/retry granularity without reducing throughput.
    """

    levels: dict[str, int] = {}
    buckets: dict[tuple[int, str], list[Any]] = {}
    stage_order: dict[int, list[str]] = {}

    for module in modules:
        missing = [dependency for dependency in module.depends_on if dependency not in levels]
        if missing:
            raise work_graph_module.WorkGraphError(
                "Module sharding requires topological order; unresolved dependencies for "
                f"{module.module_id}: {missing[:4]}"
            )
        level = (
            0
            if not module.depends_on
            else 1 + max(levels[dependency] for dependency in module.depends_on)
        )
        levels[module.module_id] = level
        stage = work_graph_module._module_stage(module)
        key = (level, stage)
        buckets.setdefault(key, []).append(module)
        stages = stage_order.setdefault(level, [])
        if stage not in stages:
            stages.append(stage)

    for level in sorted(stage_order):
        for stage in stage_order[level]:
            values = buckets[(level, stage)]
            shard_size = (
                1
                if stage == "custom"
                else policy.entity_shard_size
                if stage == "entity"
                else policy.java_shard_size
            )
            for index in range(0, len(values), shard_size):
                yield stage, tuple(values[index : index + shard_size])


def install(*, complete_planner_module: Any, work_graph_module: Any) -> None:
    """Remove proven serial critical-path waste without increasing GPU concurrency."""

    planner_cls = complete_planner_module.CompleteGameDesignPlanner
    current_expand = planner_cls._expand_one_production_batch
    if not getattr(current_expand, "_mmm_adaptive_page_width", False):
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


__all__ = ["install", "_dependency_wave_shards"]
