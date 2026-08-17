from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .central_research import normalize_research_brief, retrieve_domain_evidence
from .complete_spec import (
    AssetRequest,
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .game_design import GameDesignPlanner
from .model_router import ModelRouter
from .planner_template_schema import (
    TOP_LEVEL_KEYS,
    build_batch_skeleton,
    merge_model_output_into_skeleton,
)
from .production_contract import compile_production_contract
from .spec import SpecValidationError

# The host owns every structured shape. The model supplies values only.
_PRODUCTION_OUTLINE_CONTRACT: dict[str, Any] = {
    "production_batches": [
        {
            "batch_id": "core_features",
            "scope": "Implement the requested core features.",
            "depends_on_batches": [],
            "deliverables": ["core_features_complete"],
            "exports": ["core_features"],
        }
    ],
    "complete": True,
    "next_cursor": "",
}
_PRODUCTION_PAGE_CONTRACT: dict[str, Any] = {
    "modules": [],
    "assets": [],
    "acceptance_tests": [],
    "completed_deliverables": [],
    "complete": True,
    "next_cursor": "",
}

# Kept empty only as a stable module attribute for callers that introspect it.
# Alias translation itself is intentionally retired.
_FIELD_ALIASES: dict[str, str] = {}

# Prevent the retired runtime monkeypatch from wrapping this implementation while
# runtime_bootstrap is being collapsed to direct ownership.
_mmm_planner_json_runtime_contract = True

_SYSTEM_PROMPT = (
    "Return only the host-requested JSON shape. Do not invent fields. "
    "The host owns identifiers, validation, dependencies and completion semantics."
)


@dataclass(frozen=True)
class _ProductionBatch:
    batch_id: str
    scope: str
    depends_on_batches: tuple[str, ...]
    deliverables: tuple[str, ...]
    exports: tuple[str, ...]


class CompleteGameDesignPlanner:
    """Plan a complete mod through one closed host-owned template pipeline."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> CompleteProposal:
        session_factory = getattr(self.router, "generation_session", None)
        if not callable(session_factory):
            return self._plan_in_session(
                prompt,
                media_paths=media_paths,
                existing_input_sha256=existing_input_sha256,
            )
        with session_factory("planner"):
            return self._plan_in_session(
                prompt,
                media_paths=media_paths,
                existing_input_sha256=existing_input_sha256,
            )

    def _plan_in_session(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> CompleteProposal:
        game_design, base_proposal = GameDesignPlanner(self.router).plan(
            prompt,
            media_paths=media_paths,
        )
        research_brief = game_design.get("_research_brief")
        if not isinstance(research_brief, dict):
            research_brief = normalize_research_brief(prompt, game_design)

        internal_design = {
            **game_design,
            "_research_brief": research_brief,
            "_technical_evidence": _retrieve_implementation_evidence(
                prompt,
                game_design,
                research_brief,
            ),
        }

        batches = self._plan_batches(prompt, internal_design)
        modules, assets, acceptance_tests = self._expand_batches(
            batches,
            prompt=prompt,
            game_design=internal_design,
        )

        contract_design = {
            key: value
            for key, value in internal_design.items()
            if not str(key).startswith("_")
        }
        compiled = compile_production_contract(
            requested_prompt=prompt,
            game_design=contract_design,
            research_brief=research_brief,
            modules=modules,
            assets=assets,
            acceptance_tests=acceptance_tests,
        )
        internal_design = {
            **internal_design,
            "production_outline": [_batch_dict(batch) for batch in batches],
            "_production_contract": compiled.contract,
        }
        return complete_proposal_from_parts(
            requested_prompt=prompt,
            base_proposal=base_proposal,
            game_design=internal_design,
            modules=modules,
            assets=assets,
            acceptance_tests=tuple(compiled.acceptance_tests),
            existing_input_sha256=existing_input_sha256,
        )

    def _plan_batches(
        self,
        prompt: str,
        game_design: dict[str, Any],
    ) -> tuple[_ProductionBatch, ...]:
        request = {
            "request": prompt,
            "design": _implementation_research_outline(game_design),
            "template": _PRODUCTION_OUTLINE_CONTRACT,
        }
        try:
            page = _generate_json_page_with_repair(
                self.router,
                system_prompt=_SYSTEM_PROMPT,
                request=request,
                media_paths=(),
                expected_contracts=(frozenset(_PRODUCTION_OUTLINE_CONTRACT),),
                stage="production outline",
            )
        except (SpecValidationError, ValueError, TypeError, RuntimeError):
            page = _fallback_outline(prompt)

        batches = _validated_batches(page.get("production_batches"))
        if not batches:
            batches = _validated_batches(_fallback_outline(prompt)["production_batches"])
        return _topological_batches(batches)

    def _expand_batches(
        self,
        batches: Sequence[_ProductionBatch],
        *,
        prompt: str,
        game_design: dict[str, Any],
    ) -> tuple[tuple[ProductionModule, ...], tuple[AssetRequest, ...], tuple[str, ...]]:
        modules: list[ProductionModule] = []
        assets: list[AssetRequest] = []
        tests: list[str] = []
        known_module_ids: set[str] = set()
        exports_by_batch: dict[str, tuple[str, ...]] = {}

        for batch in batches:
            dependency_ids = tuple(
                module_id
                for dependency in batch.depends_on_batches
                for module_id in exports_by_batch.get(dependency, ())
            )
            skeleton = build_batch_skeleton(
                batch_id=batch.batch_id,
                scope=batch.scope,
                deliverables=batch.deliverables,
                exports=batch.exports,
                depends_on_batches=dependency_ids,
                known_module_ids=tuple(known_module_ids),
            )
            request = {
                "request": prompt,
                "batch": _batch_dict(batch),
                "design": _implementation_research_outline(game_design),
                "template_skeleton": skeleton,
                "contract": _PRODUCTION_PAGE_CONTRACT,
            }
            try:
                raw_page = _generate_json_page_with_repair(
                    self.router,
                    system_prompt=(
                        "Fill values inside template_skeleton only. Keep every host-owned "
                        "field name and identifier. Unknown fields are discarded."
                    ),
                    request=request,
                    media_paths=(),
                    expected_contracts=(frozenset(TOP_LEVEL_KEYS),),
                    stage=f"production batch {batch.batch_id}",
                )
            except (SpecValidationError, ValueError, TypeError, RuntimeError):
                raw_page = {}

            page = merge_model_output_into_skeleton(
                skeleton=skeleton,
                model_output=raw_page,
                valid_module_catalog=known_module_ids | set(batch.exports),
            )
            expected_ids = {
                str(item["module_id"])
                for item in skeleton["modules"]
                if isinstance(item, dict) and item.get("module_id")
            }
            accepted_modules = [
                _module(raw)
                for raw in page["modules"]
                if isinstance(raw, dict)
                and str(raw.get("module_id") or "") in expected_ids
                and str(raw.get("module_id") or "") not in known_module_ids
            ]
            if not accepted_modules:
                accepted_modules = [_module(raw) for raw in skeleton["modules"]]

            for module in accepted_modules:
                modules.append(module)
                known_module_ids.add(module.module_id)

            for raw in page["assets"]:
                if not isinstance(raw, dict):
                    continue
                asset = _asset(raw)
                if asset.asset_id not in {item.asset_id for item in assets} and asset.target_path not in {
                    item.target_path for item in assets
                }:
                    assets.append(asset)

            tests.extend(_unique_strings(page.get("acceptance_tests")))
            exports_by_batch[batch.batch_id] = tuple(
                module.module_id for module in accepted_modules
            )

        if not modules:
            fallback = build_batch_skeleton(
                batch_id="core_features",
                scope="Implement the requested core features.",
                deliverables=("core_features_complete",),
                exports=("core_features",),
            )
            modules = [_module(raw) for raw in fallback["modules"]]
            tests.extend(_unique_strings(fallback["acceptance_tests"]))

        tests = list(dict.fromkeys(test for test in tests if test))
        if not tests:
            tests = [f"test_{module.module_id}_registers" for module in modules]
        return tuple(modules), tuple(assets), tuple(tests)


def _implementation_prompt(prompt: str, game_design: dict[str, Any]) -> str:
    """Return the semantic implementation request; platform coordinates stay host-owned."""
    design = json.dumps(_implementation_research_outline(game_design), ensure_ascii=False)
    return (
        "Implement the requested Minecraft mod features through the host-owned production "
        f"template. Request: {prompt}\nDesign context: {design}"
    )


def _implementation_research_outline(game_design: dict[str, Any]) -> dict[str, Any]:
    """Expose bounded planner context without copying runtime-only decoration."""
    keys = (
        "mod_id",
        "mod_name",
        "description",
        "features",
        "systems",
        "constraints",
        "acceptance_tests",
        "_platform_selection",
        "_platform_evidence",
        "_research_brief",
        "_technical_evidence",
        "_pre_design_research",
    )
    return {key: game_design[key] for key in keys if key in game_design}


def _retrieve_implementation_evidence(
    prompt: str,
    game_design: dict[str, Any],
    research_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = game_design.get("_platform_evidence")
    if isinstance(existing, Mapping):
        return dict(existing)
    brief = research_brief or normalize_research_brief(prompt, game_design)
    try:
        value = retrieve_domain_evidence(brief)
    except (SpecValidationError, ValueError, TypeError, RuntimeError):
        return {
            "schema_version": "mmm/research-unavailable-v1",
            "domains": [],
            "status": "unavailable",
        }
    return dict(value) if isinstance(value, Mapping) else {
        "schema_version": "mmm/research-unavailable-v1",
        "domains": [],
        "status": "unavailable",
    }


def _generate_json_page_with_repair(
    router: Any,
    *,
    system_prompt: str,
    request: dict[str, Any] | str,
    media_paths: Sequence[str | Path],
    expected_contracts: Sequence[frozenset[str]],
    stage: str,
) -> dict[str, Any]:
    """Generate one page once; callers own deterministic host fallback.

    Previous page-local repair loops are intentionally removed. A malformed model
    page never mutates the contract and never triggers unbounded model replanning.
    """
    request_text = request if isinstance(request, str) else json.dumps(request, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request_text},
    ]
    kwargs = {
        "media_paths": media_paths,
        "response_format": "json",
    }
    try:
        text = router.generate_text(
            "planner",
            messages,
            enable_tools=False,
            **kwargs,
        )
    except TypeError:
        text = router.generate_text("planner", messages, **kwargs)
    try:
        return _extract_json(str(text), expected_contracts=expected_contracts)
    except SpecValidationError as exc:
        raise SpecValidationError(f"{stage}: {exc}") from exc


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = start + max(end, 1)
    return objects


def _extract_json(
    text: str,
    *,
    expected_contracts: Sequence[frozenset[str]],
) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise SpecValidationError("Structured planner returned empty JSON.")
    objects = _json_objects(text)
    matches = [
        value
        for value in objects
        if frozenset(str(key) for key in value) in expected_contracts
    ]
    if not matches:
        raise SpecValidationError("Structured planner did not return the host-owned JSON contract.")

    outline_fields = frozenset(_PRODUCTION_OUTLINE_CONTRACT)
    if len(expected_contracts) == 1 and expected_contracts[0] == outline_fields:
        outline_matches = [value for value in matches if frozenset(value) == outline_fields]
        if not outline_matches:
            raise SpecValidationError("Production outline did not match the host contract.")
        batches: list[Any] = []
        for page in outline_matches:
            raw_batches = page.get("production_batches")
            if not isinstance(raw_batches, list):
                raise SpecValidationError("production_batches must be a JSON list.")
            batches.extend(raw_batches)
        terminal = outline_matches[-1]
        return {
            "production_batches": batches,
            "complete": bool(terminal.get("complete")),
            "next_cursor": str(terminal.get("next_cursor") or ""),
        }

    if len(matches) != 1:
        raise SpecValidationError("Structured planner returned ambiguous contract objects.")
    return dict(matches[0])


def _validated_batches(value: Any) -> tuple[_ProductionBatch, ...]:
    if not isinstance(value, list):
        return ()
    result: list[_ProductionBatch] = []
    seen: set[str] = set()
    for raw in value:
        try:
            batch = _production_batch(raw)
        except (SpecValidationError, ValueError, TypeError):
            continue
        if batch.batch_id in seen:
            continue
        seen.add(batch.batch_id)
        result.append(batch)
    return tuple(result)


def _production_batch(value: Any) -> _ProductionBatch:
    if not isinstance(value, Mapping):
        raise SpecValidationError("Production batch must be an object.")
    allowed = {
        "batch_id",
        "scope",
        "depends_on_batches",
        "deliverables",
        "exports",
    }
    if set(value) != allowed:
        raise SpecValidationError("Production batch fields do not match the host contract.")
    batch_id = _identifier(value.get("batch_id"), "batch")
    scope = str(value.get("scope") or "").strip()
    if not scope:
        raise SpecValidationError("Production batch scope is empty.")
    deliverables = tuple(_unique_strings(value.get("deliverables")))
    exports = tuple(_identifier(item, "module") for item in _unique_strings(value.get("exports")))
    if not deliverables or not exports:
        raise SpecValidationError("Production batch must declare deliverables and exports.")
    dependencies = tuple(_identifier(item, "batch") for item in _unique_strings(value.get("depends_on_batches")))
    if batch_id in dependencies:
        raise SpecValidationError("Production batch may not depend on itself.")
    return _ProductionBatch(
        batch_id=batch_id,
        scope=scope,
        depends_on_batches=dependencies,
        deliverables=deliverables,
        exports=exports,
    )


def _topological_batches(batches: Sequence[_ProductionBatch]) -> tuple[_ProductionBatch, ...]:
    by_id = {batch.batch_id: batch for batch in batches}
    pending = list(batches)
    emitted: list[_ProductionBatch] = []
    emitted_ids: set[str] = set()
    while pending:
        progress = False
        next_pending: list[_ProductionBatch] = []
        for batch in pending:
            known_dependencies = tuple(
                dependency
                for dependency in batch.depends_on_batches
                if dependency in by_id
            )
            if set(known_dependencies) <= emitted_ids:
                emitted.append(
                    _ProductionBatch(
                        batch_id=batch.batch_id,
                        scope=batch.scope,
                        depends_on_batches=known_dependencies,
                        deliverables=batch.deliverables,
                        exports=batch.exports,
                    )
                )
                emitted_ids.add(batch.batch_id)
                progress = True
            else:
                next_pending.append(batch)
        if not progress:
            # Cyclic model dependencies are discarded rather than patched at runtime.
            for batch in next_pending:
                emitted.append(
                    _ProductionBatch(
                        batch_id=batch.batch_id,
                        scope=batch.scope,
                        depends_on_batches=(),
                        deliverables=batch.deliverables,
                        exports=batch.exports,
                    )
                )
            break
        pending = next_pending
    return tuple(emitted)


def _module(value: Mapping[str, Any]) -> ProductionModule:
    return ProductionModule(
        module_id=str(value["module_id"]),
        kind=str(value.get("kind") or "custom_java"),
        config=dict(value.get("config") or {}),
        depends_on=tuple(_unique_strings(value.get("depends_on"))),
        required_gates=tuple(_unique_strings(value.get("required_gates"))),
    )


def _asset(value: Mapping[str, Any]) -> AssetRequest:
    return AssetRequest(
        asset_id=str(value["asset_id"]),
        kind=str(value["kind"]),
        prompt=str(value["prompt"]),
        target_path=str(value["target_path"]),
        width=int(value.get("width", 16)),
        height=int(value.get("height", 16)),
    )


def _fallback_outline(prompt: str) -> dict[str, Any]:
    summary = " ".join(str(prompt).strip().split())[:240] or "requested mod features"
    return {
        "production_batches": [
            {
                "batch_id": "core_features",
                "scope": f"Implement the complete requested mod behavior: {summary}",
                "depends_on_batches": [],
                "deliverables": ["requested_mod_behavior_complete"],
                "exports": ["core_features"],
            }
        ],
        "complete": True,
        "next_cursor": "",
    }


def _identifier(value: Any, fallback: str) -> str:
    import re

    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not text or not text[0].isalpha():
        text = f"{fallback}_{text}".rstrip("_")
    return text[:63]


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    )


def _batch_dict(batch: _ProductionBatch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "scope": batch.scope,
        "depends_on_batches": list(batch.depends_on_batches),
        "deliverables": list(batch.deliverables),
        "exports": list(batch.exports),
    }


__all__ = ["CompleteGameDesignPlanner"]
