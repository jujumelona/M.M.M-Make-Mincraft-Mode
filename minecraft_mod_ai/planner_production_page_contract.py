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


class _StagedCatalog:
    """Overlay catalog used while a production page is still uncommitted.

    Child-item repair needs duplicate detection across both prior accepted output and
    siblings from the current page. Mutating the real catalog during that validation
    makes a later page-level failure irreversible because the catalog digest is
    append-only. This overlay records current-page identities locally and publishes
    nothing until the whole page has passed its progress checks.
    """

    def __init__(self, base: Any) -> None:
        self._base = base
        self._added: set[str] = set()

    def __contains__(self, value: str) -> bool:
        return value in self._added or value in self._base

    def add(self, value: str) -> None:
        if value in self:
            raise ValueError(f"duplicate staged production id: {value}")
        self._added.add(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if isinstance(item, str) and str(item).strip()
    ]


def _upgrade_legacy_evidence_page(
    page: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Upgrade only the immediately previous durable page schema in memory.

    Current structured generation receives the live contract and therefore must emit
    ``deliverable_evidence``. A cached page written before that field was introduced
    can still be resumed safely under its previously accepted completion semantics.
    No other missing or extra field is normalized here.
    """

    expected = set(contract)
    legacy = expected - {"deliverable_evidence"}
    if (
        "deliverable_evidence" in expected
        and "deliverable_evidence" not in page
        and set(page) == legacy
    ):
        upgraded = dict(page)
        upgraded["deliverable_evidence"] = {}
        return upgraded
    return page


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
        from .planner_structured_router import structured_planner_router
        from .production_page_durable_contract import (
            load_or_generate_page,
            resolve_page_items,
        )

        remaining = list(dict.fromkeys(str(value) for value in batch.deliverables))
        cursor = ""
        first_page = True
        seen_states: set[tuple[tuple[str, ...], str]] = set()
        structured_router = structured_planner_router(self.router)

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
                # The host already supplied the exact evidence/context and a strict
                # response schema. Tool use here only adds serial model-tool-model
                # round-trips, so structured production decode is deliberately direct.
                return complete_planner_module._generate_json_page_with_repair(
                    structured_router,
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
            page = _upgrade_legacy_evidence_page(
                page,
                complete_planner_module._PRODUCTION_PAGE_CONTRACT,
            )

            if set(page) != set(complete_planner_module._PRODUCTION_PAGE_CONTRACT):
                raise complete_planner_module.SpecValidationError(
                    "Production batch page fields are invalid."
                )

            # A page with no host target progress is rejected before any child parser or
            # repair path can mutate proposal/catalog state.
            completed = {
                value
                for value in _string_list(page.get("completed_deliverables", []))
                if value in remaining
            }
            if not completed:
                raise complete_planner_module.SpecValidationError(
                    f"Production batch {batch.batch_id!r} page made no verified progress."
                )

            # Resolve children against staged catalog overlays. Semantic child repair
            # also uses the direct structured router: it operates only on the persisted
            # invalid child plus validator error and does not need another RAG cycle.
            staged_modules = _StagedCatalog(module_catalog)
            staged_assets = _StagedCatalog(asset_catalog)
            staged_audio = _StagedCatalog(audio_catalog)
            page_modules, page_assets, page_audio, tests = resolve_page_items(
                complete_planner_module,
                structured_router,
                page=page,
                page_path=page_path,
                module_catalog=staged_modules,
                asset_catalog=staged_assets,
                audio_catalog=staged_audio,
                test_catalog=test_catalog,
            )

            # Publish catalog identities and proposal output only after the complete
            # page has passed structural, progress, and child-item validation.
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
    expand_one_production_batch._mmm_structured_no_tool_loop = True  # type: ignore[attr-defined]
    cls._expand_one_production_batch = expand_one_production_batch


__all__ = ["install"]
