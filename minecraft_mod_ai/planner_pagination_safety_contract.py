from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def _outline_state(module: Any, page: dict[str, Any]) -> tuple[list[Any], bool, str]:
    if not isinstance(page, dict):
        return [], True, ""
    raw_batches = page.get("production_batches")
    if not isinstance(raw_batches, list):
        raw_batches = []
    complete = bool(page.get("complete", True))
    next_cursor = str(page.get("next_cursor") or "").strip()
    if complete:
        next_cursor = ""
    elif not next_cursor:
        complete = True
    return raw_batches, complete, next_cursor


def _append_outline_batches(
    module: Any,
    *,
    raw_batches: list[Any],
    catalog: Any,
    result: list[Any],
) -> None:
    for raw in raw_batches:
        if not isinstance(raw, dict):
            raise module.SpecValidationError(
                "Production outline batch must be a JSON object."
            )
        try:
            batch = module._production_batch(raw)
        except Exception as exc:
            raise module.SpecValidationError(
                f"Production outline contains an invalid batch: {exc}"
            ) from exc
        original_id = batch.batch_id
        suffix = 2
        while batch.batch_id in catalog:
            batch = module._ProductionBatch(
                batch_id=f"{original_id}_{suffix}",
                scope=batch.scope,
                depends_on_batches=batch.depends_on_batches,
                deliverables=batch.deliverables,
                exports=batch.exports,
            )
            suffix += 1
        catalog.add(batch.batch_id)
        result.append(batch)


def _advance_outline_cursor(
    module: Any,
    *,
    complete: bool,
    next_cursor: str,
    cursor: str,
    seen_cursors: set[str],
) -> str | None:
    if complete:
        return None
    if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
        return None
    seen_cursors.add(next_cursor)
    return next_cursor


def _collect_one_request_page_outline(
    self: Any,
    *,
    first_page: dict[str, Any],
    base_request: dict[str, Any],
    page_index: int,
    page_count: int,
):
    module = _planner_module(self)
    catalog = module._ModuleCatalog()
    result: list[Any] = []
    page = first_page
    cursor = ""
    seen_cursors: set[str] = set()
    while True:
        raw_batches, complete, next_cursor = _outline_state(module, page)
        _append_outline_batches(
            module,
            raw_batches=raw_batches,
            catalog=catalog,
            result=result,
        )
        next_value = _advance_outline_cursor(
            module,
            complete=complete,
            next_cursor=next_cursor,
            cursor=cursor,
            seen_cursors=seen_cursors,
        )
        if next_value is None:
            break
        cursor = next_value
        continuation_request = {
            **base_request,
            "known_local_batch_catalog": catalog.receipt(),
            "cursor": cursor,
        }
        page = module._generate_json_page_with_repair(
            self.router,
            system_prompt=module._SHARDED_REQUEST_OUTLINE_SYSTEM_PROMPT,
            request=continuation_request,
            media_paths=(),
            expected_contracts=(frozenset(module._PRODUCTION_OUTLINE_CONTRACT),),
            stage=(
                "authoritative request production outline continuation "
                f"{page_index + 1}/{page_count}"
            ),
        )
    return module._topological_production_batches(tuple(result))


def _collect_production_batches(
    self: Any,
    *,
    first_page: dict[str, Any],
    prompt: str,
    game_design: dict[str, Any],
    media_paths: Sequence[str | Path],
):
    module = _planner_module(self)
    del media_paths
    context, context_receipt = module._pagination_planning_context(prompt, game_design)
    catalog = module._ModuleCatalog()
    result: list[Any] = []
    page = first_page
    cursor = ""
    seen_cursors: set[str] = set()
    while True:
        raw_batches, complete, next_cursor = _outline_state(module, page)
        _append_outline_batches(
            module,
            raw_batches=raw_batches,
            catalog=catalog,
            result=result,
        )
        next_value = _advance_outline_cursor(
            module,
            complete=complete,
            next_cursor=next_cursor,
            cursor=cursor,
            seen_cursors=seen_cursors,
        )
        if next_value is None:
            break
        cursor = next_value
        existing_ids = sorted(
            catalog._ids
            if hasattr(catalog, "_ids")
            else (batch.batch_id for batch in result)
        )
        request = {
            "planning_context": context,
            "planning_context_receipt": context_receipt,
            "known_batch_catalog": catalog.receipt(),
            "already_generated_batch_ids": existing_ids,
            "cursor": cursor,
            "contract": module._PRODUCTION_OUTLINE_CONTRACT,
        }
        page = module._generate_json_page_with_repair(
            self.router,
            system_prompt=(
                "Continue the production outline. Return exactly one JSON object "
                "containing production_batches, complete, and next_cursor. Generate "
                "only NEW production batches. If more remain, complete=false requires "
                "a new non-empty cursor. Never repeat an existing batch ID."
            ),
            request=request,
            media_paths=(),
            expected_contracts=(frozenset(module._PRODUCTION_OUTLINE_CONTRACT),),
            stage="production outline continuation",
        )
    if not result:
        raise module.SpecValidationError("Production outline generated zero batches.")
    return module._topological_production_batches(tuple(result))


def _page_outputs(module: Any, page: dict[str, Any], test_catalog: set[str]):
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
    tests = [str(value).strip() for value in raw_tests if str(value).strip()]
    tests = [value for value in tests if value not in test_catalog]
    return raw_modules, raw_assets, raw_audio, page_modules, page_assets, page_audio, tests


def _verified_completed_targets(
    page: dict[str, Any],
    *,
    target_deliverables: Sequence[str],
    raw_modules: Sequence[Any],
    raw_assets: Sequence[Any],
    raw_audio: Sequence[Any],
    page_modules: Sequence[Any],
    page_assets: Sequence[Any],
    page_audio: Sequence[Any],
    tests: Sequence[str],
) -> set[str]:
    completed = page.get("completed_deliverables")
    if not isinstance(completed, list):
        return set()
    completed_names = {
        str(value).strip()
        for value in completed
        if isinstance(value, str) and str(value).strip()
    }
    targets = set(target_deliverables)
    if completed_names - targets:
        return set()

    page_mod_ids = {item.module_id for item in page_modules}
    page_asset_ids = {item.asset_id for item in page_assets}
    page_audio_ids = {item.sound_id for item in page_audio}
    page_tests = set(tests)
    completed_set: set[str] = set()

    for raw_item in (*raw_modules, *raw_assets, *raw_audio):
        if not isinstance(raw_item, dict):
            continue
        item_id = str(
            raw_item.get("module_id")
            or raw_item.get("asset_id")
            or raw_item.get("sound_id")
            or ""
        ).strip()
        claims = raw_item.get("implements_deliverables") or raw_item.get("implements") or ()
        if isinstance(claims, (list, tuple)) and item_id:
            for claim in claims:
                value = str(claim).strip()
                if value in targets:
                    completed_set.add(value)
        if item_id in targets:
            completed_set.add(item_id)

    produced_any = bool(page_mod_ids or page_asset_ids or page_audio_ids or page_tests)
    if produced_any:
        for deliverable in target_deliverables:
            if deliverable in page_mod_ids or deliverable in page_asset_ids:
                completed_set.add(deliverable)
            elif deliverable in page_audio_ids or deliverable in page_tests:
                completed_set.add(deliverable)
            elif deliverable in completed_names:
                # The host schema has already constrained completed_deliverables to
                # the current target. Preserve the legacy fallback, but only when
                # this page produced concrete implementation/test output.
                completed_set.add(deliverable)
    return completed_set


def _expand_one_production_batch(
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
    media_paths: Sequence[str | Path],
) -> None:
    module = _planner_module(self)
    remaining = list(batch.deliverables)
    cursor = ""
    seen_cursors: set[str] = set()
    first_page = True

    while remaining:
        target_deliverables = remaining[: min(len(remaining), 4)]
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
                "Return exactly one production-batch JSON page. Implement ALL current "
                "target deliverables, list only genuinely completed target names in "
                "completed_deliverables, and use a new non-empty next_cursor whenever "
                "more host deliverables remain. Never repeat an ID or file path."
            ),
            request=request,
            media_paths=media_paths if first_page else (),
            expected_contracts=(frozenset(module._PRODUCTION_PAGE_CONTRACT),),
            stage=f"production batch {batch.batch_id!r} page",
        )
        first_page = False
        if set(page) != set(module._PRODUCTION_PAGE_CONTRACT):
            raise module.SpecValidationError("Production batch page fields are invalid.")
        complete = page.get("complete")
        next_cursor = page.get("next_cursor")
        if type(complete) is not bool or not isinstance(next_cursor, str):
            raise module.SpecValidationError("Production batch pagination contract is invalid.")

        (
            raw_modules,
            raw_assets,
            raw_audio,
            page_modules,
            page_assets,
            page_audio,
            tests,
        ) = _page_outputs(module, page, test_catalog)
        completed_set = _verified_completed_targets(
            page,
            target_deliverables=target_deliverables,
            raw_modules=raw_modules,
            raw_assets=raw_assets,
            raw_audio=raw_audio,
            page_modules=page_modules,
            page_assets=page_assets,
            page_audio=page_audio,
            tests=tests,
        )
        if not completed_set:
            raise module.SpecValidationError(
                "Production batch page made no host-verifiable deliverable progress."
            )

        previous_remaining = tuple(remaining)
        remaining = [value for value in remaining if value not in completed_set]
        if len(remaining) >= len(previous_remaining):
            raise module.SpecValidationError(
                "Production batch pagination did not reduce remaining deliverables."
            )

        if remaining:
            if complete:
                raise module.SpecValidationError(
                    "Production batch declared complete with deliverables still remaining."
                )
            if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
                raise module.SpecValidationError(
                    "Production batch pagination did not advance its cursor."
                )
        else:
            if not complete:
                raise module.SpecValidationError(
                    "Production batch completed all deliverables but complete=false."
                )
            if next_cursor:
                raise module.SpecValidationError(
                    "Complete production batch page may not have next_cursor."
                )

        # Commit catalog and proposal mutations only after the page has passed both
        # progress and cursor checks. A rejected page therefore cannot contaminate
        # subsequent planner context.
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

        if remaining:
            seen_cursors.add(next_cursor)
            cursor = next_cursor


def _planner_module(self: Any) -> Any:
    import sys

    return sys.modules[self.__class__.__module__]


def install(complete_planner_module: Any) -> None:
    """Replace planner pagination paths that could silently stall or loop forever."""

    cls = complete_planner_module.CompleteGameDesignPlanner
    if getattr(cls, "_mmm_pagination_safety_contract", False):
        return
    cls._collect_one_request_page_outline = _collect_one_request_page_outline
    cls._collect_production_batches = _collect_production_batches
    cls._expand_one_production_batch = _expand_one_production_batch
    cls._mmm_pagination_safety_contract = True


__all__ = [
    "install",
]
