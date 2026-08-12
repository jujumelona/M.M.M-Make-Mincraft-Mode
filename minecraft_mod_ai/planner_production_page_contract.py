from __future__ import annotations

from functools import wraps
from typing import Any, Sequence


_ADAPTIVE_PRODUCTION_PROMPT = """Return exactly one production-batch JSON page.

The host supplies current_target_deliverables as the COMPLETE unresolved pool for this
batch. There is NO fixed deliverable count per response and NO fixed page count.
Choose the largest coherent subset you can FULLY implement within the available output
budget without truncating JSON, code metadata, assets, audio, or tests. Do not pad or
artificially split work merely to create another page.

A deliverable may require multiple modules/assets/tests and one output may satisfy
multiple tightly coupled deliverables. Do NOT force one module per deliverable.
Whenever practical, add implements_deliverables to produced module/asset/audio objects
using exact names from current_target_deliverables. Put ONLY actually completed exact
names in completed_deliverables. Never claim a partially emitted deliverable.

If every unresolved deliverable is fully completed, set complete=true and
next_cursor="". Otherwise set complete=false and return a short non-empty next_cursor.
Never repeat an already-known module, asset, audio ID, or file path. Return JSON only.
""".strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if isinstance(item, str) and str(item).strip()
    ]


def _append_unique_parsed(
    *,
    raw_items: Sequence[Any],
    parser: Any,
    catalog: Any,
    id_attr: str,
    destination: list[Any],
) -> None:
    """Preserve valid siblings when one generated item is malformed or duplicated."""

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            parsed = parser(raw)
            identifier = str(getattr(parsed, id_attr)).strip()
        except Exception:
            continue
        if not identifier or identifier in catalog:
            continue
        try:
            catalog.add(identifier)
        except Exception:
            continue
        destination.append(parsed)


def install(complete_planner_module: Any) -> None:
    """Let the model choose production-page width while the host owns progress."""

    cls = complete_planner_module.CompleteGameDesignPlanner
    current = cls._expand_one_production_batch
    if getattr(current, "_mmm_adaptive_production_page_width", False):
        return

    @wraps(current)
    def expand_one_production_batch(
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
        remaining = list(dict.fromkeys(str(value) for value in batch.deliverables))
        cursor = ""
        seen_cursors: set[str] = set()
        first_page = True

        while remaining:
            # The unresolved pool, not an arbitrary host slice, is the target. The
            # model decides how much of it fits safely in this response.
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
                "contract": complete_planner_module._PRODUCTION_PAGE_CONTRACT,
            }
            if first_page:
                request["planning_context"] = planning_context

            page = complete_planner_module._generate_json_page_with_repair(
                self.router,
                system_prompt=_ADAPTIVE_PRODUCTION_PROMPT,
                request=request,
                media_paths=media_paths if first_page else (),
                expected_contracts=(
                    frozenset(complete_planner_module._PRODUCTION_PAGE_CONTRACT),
                ),
                stage=f"production batch {batch.batch_id!r} page",
            )
            first_page = False

            raw_modules = page.get("modules", [])
            raw_assets = page.get("assets", [])
            raw_audio = page.get("audio", [])
            raw_tests = page.get("acceptance_tests", [])
            if not isinstance(raw_modules, list):
                raw_modules = []
            if not isinstance(raw_assets, list):
                raw_assets = []
            if not isinstance(raw_audio, list):
                raw_audio = []
            if not isinstance(raw_tests, list):
                raw_tests = []

            # Parse each item independently. A bad sibling is not a reason to discard
            # valid output already generated in the same page.
            _append_unique_parsed(
                raw_items=raw_modules,
                parser=complete_planner_module._module,
                catalog=module_catalog,
                id_attr="module_id",
                destination=parts.modules,
            )
            _append_unique_parsed(
                raw_items=raw_assets,
                parser=complete_planner_module._asset,
                catalog=asset_catalog,
                id_attr="asset_id",
                destination=parts.assets,
            )
            _append_unique_parsed(
                raw_items=raw_audio,
                parser=complete_planner_module._audio,
                catalog=audio_catalog,
                id_attr="sound_id",
                destination=parts.audio,
            )

            tests = _string_list(raw_tests)
            for test in tests:
                if test in test_catalog:
                    continue
                test_catalog.add(test)
                parts.acceptance_tests.append(test)

            completed = [
                value
                for value in _string_list(page.get("completed_deliverables", []))
                if value in remaining
            ]
            completed_set = set(completed)
            if not completed_set:
                # planner_json_runtime_contract normally rejects this before return.
                # Keep a fail-closed fence here so this loop can never spin forever if
                # another future wrapper changes that validation behavior.
                raise complete_planner_module.SpecValidationError(
                    "Production page made no host-verifiable deliverable progress."
                )

            remaining = [
                value for value in remaining if value not in completed_set
            ]
            if not remaining:
                break

            next_cursor = page.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                # Host progress, not model memory, is authoritative. A deterministic
                # continuation token is sufficient because the full remaining pool is
                # re-sent on every call.
                next_cursor = f"host_remaining_{len(remaining)}"
            if next_cursor in seen_cursors:
                next_cursor = f"host_remaining_{len(remaining)}"
            if next_cursor in seen_cursors:
                raise complete_planner_module.SpecValidationError(
                    "Production pagination cursor stalled despite unresolved work."
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    expand_one_production_batch._mmm_adaptive_production_page_width = True  # type: ignore[attr-defined]
    cls._expand_one_production_batch = expand_one_production_batch


__all__ = ["install"]
