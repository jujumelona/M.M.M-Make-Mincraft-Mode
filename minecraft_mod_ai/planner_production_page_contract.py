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


def install(complete_planner_module: Any) -> None:
    """Make model-owned page width the final durable production-page policy."""

    cls = complete_planner_module.CompleteGameDesignPlanner
    current = cls._expand_one_production_batch
    if getattr(current, "_mmm_adaptive_production_page_width", False) and getattr(
        current,
        "_mmm_durable_production_items",
        False,
    ):
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
        from .production_page_durable_contract import (
            load_or_generate_page,
            resolve_page_items,
        )

        remaining = list(dict.fromkeys(str(value) for value in batch.deliverables))
        cursor = ""
        first_page = True
        seen_states: set[tuple[tuple[str, ...], str]] = set()

        while remaining:
            state = (tuple(remaining), cursor)
            if state in seen_states:
                raise complete_planner_module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} pagination made no progress."
                )
            seen_states.add(state)

            # Never slice the unresolved pool to an arbitrary host width. The model
            # chooses how many coherent deliverables fit in this response; the host
            # validates and persists the exact completed subset.
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

            stage = f"production batch {batch.batch_id!r} page"

            def generate_page() -> dict[str, Any]:
                return complete_planner_module._generate_json_page_with_repair(
                    self.router,
                    system_prompt=_ADAPTIVE_PRODUCTION_PROMPT,
                    request=request,
                    media_paths=media_paths if first_page else (),
                    expected_contracts=(
                        frozenset(complete_planner_module._PRODUCTION_PAGE_CONTRACT),
                    ),
                    stage=stage,
                )

            # The page is durably keyed by the exact host request. A process/backend
            # interruption after a successful decode therefore resumes from disk rather
            # than spending another full GPU generation on already accepted output.
            page, page_path = load_or_generate_page(
                stage=stage,
                request=request,
                generate=generate_page,
            )
            first_page = False

            if set(page) != set(complete_planner_module._PRODUCTION_PAGE_CONTRACT):
                raise complete_planner_module.SpecValidationError(
                    "Production batch page fields are invalid."
                )

            # Parse and repair each child independently. Valid siblings are committed
            # unchanged; one malformed module/asset/audio item is patched in place and
            # never causes the whole production page to be regenerated.
            page_modules, page_assets, page_audio, tests = resolve_page_items(
                complete_planner_module,
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

            completed = {
                value
                for value in _string_list(page.get("completed_deliverables", []))
                if value in remaining
            }
            if not completed:
                raise complete_planner_module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} page made no verified progress."
                )

            remaining = [value for value in remaining if value not in completed]
            if not remaining:
                break

            next_cursor = page.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                # Continuation does not depend on opaque model memory because the exact
                # unresolved pool is resent every round.
                next_cursor = f"host_remaining_{len(remaining)}"
            cursor = next_cursor

    expand_one_production_batch._mmm_adaptive_production_page_width = True  # type: ignore[attr-defined]
    expand_one_production_batch._mmm_adaptive_page_width = True  # type: ignore[attr-defined]
    expand_one_production_batch._mmm_durable_production_items = True  # type: ignore[attr-defined]
    cls._expand_one_production_batch = expand_one_production_batch


__all__ = ["install"]
