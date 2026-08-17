from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import central_research
from .complete_spec import (
    AssetRequest,
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .game_design import GameDesignPlanner
from .model_router import ModelRouter
from .planner_template_schema import build_batch_skeleton, merge_model_output_into_skeleton
from .production_contract import compile_production_contract
from .spec import SpecValidationError

_SYSTEM_PROMPT = (
    "Fill values inside the supplied host template only. "
    "The host owns identifiers, dependencies, completion semantics, and the final schema. "
    "Do not invent top-level fields or identifiers."
)


@dataclass(frozen=True)
class _ProductionBatch:
    batch_id: str
    scope: str
    depends_on_batches: tuple[str, ...]
    deliverables: tuple[str, ...]
    exports: tuple[str, ...]


class CompleteGameDesignPlanner:
    """Plan production through deterministic host-owned batch templates.

    The model never creates the batch graph or schema. Host code derives the complete
    production template from the validated game design, creates every module identity,
    and merges only allowed values from one model response. Invalid, partial, or missing
    model output falls back to the unchanged host skeleton instead of triggering a
    repair/replan loop.
    """

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
            research_brief = central_research.normalize_research_brief(prompt, game_design)

        internal_design = {
            **game_design,
            "_research_brief": research_brief,
            "_technical_evidence": _retrieve_implementation_evidence(
                prompt,
                game_design,
                research_brief,
            ),
        }

        batches = _host_batches(prompt, internal_design)
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
            }
            raw_page = _generate_json_page(
                self.router,
                system_prompt=_SYSTEM_PROMPT,
                request=request,
                media_paths=(),
            )
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

            known_asset_ids = {item.asset_id for item in assets}
            known_asset_paths = {item.target_path for item in assets}
            for raw in page["assets"]:
                if not isinstance(raw, dict):
                    continue
                asset = _asset(raw)
                if asset.asset_id in known_asset_ids or asset.target_path in known_asset_paths:
                    continue
                assets.append(asset)
                known_asset_ids.add(asset.asset_id)
                known_asset_paths.add(asset.target_path)

            tests.extend(_unique_strings(page.get("acceptance_tests")))
            exports_by_batch[batch.batch_id] = tuple(
                module.module_id for module in accepted_modules
            )

        if not modules:
            fallback = build_batch_skeleton(
                batch_id="core_features",
                scope="Implement the complete requested mod behavior.",
                deliverables=("requested_mod_behavior_complete",),
                exports=("core_features",),
            )
            modules = [_module(raw) for raw in fallback["modules"]]
            tests.extend(_unique_strings(fallback["acceptance_tests"]))

        tests = list(dict.fromkeys(test for test in tests if test))
        if not tests:
            tests = [f"test_{module.module_id}_registers" for module in modules]
        return tuple(modules), tuple(assets), tuple(tests)


def _host_batches(prompt: str, game_design: Mapping[str, Any]) -> tuple[_ProductionBatch, ...]:
    """Build one complete production template without model-owned batch structure."""
    raw_modules = game_design.get("modules")
    exports: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_modules, list):
        for index, raw in enumerate(raw_modules):
            if not isinstance(raw, Mapping):
                continue
            module_id = _identifier(raw.get("plugin_id"), f"feature_{index + 1}")
            if module_id in seen:
                continue
            seen.add(module_id)
            exports.append(module_id)

    if exports:
        catalog = ", ".join(exports)
        return (
            _ProductionBatch(
                batch_id="requested_features",
                scope=(
                    "Implement every requested capability in this host-owned production "
                    f"template: {catalog}."
                ),
                depends_on_batches=(),
                deliverables=tuple(f"{module_id}_complete" for module_id in exports),
                exports=tuple(exports),
            ),
        )

    summary = " ".join(str(prompt).strip().split())[:240] or "requested mod features"
    return (
        _ProductionBatch(
            batch_id="core_features",
            scope=f"Implement the complete requested mod behavior: {summary}",
            depends_on_batches=(),
            deliverables=("requested_mod_behavior_complete",),
            exports=("core_features",),
        ),
    )


def _implementation_prompt(prompt: str, game_design: dict[str, Any]) -> str:
    design = json.dumps(_implementation_research_outline(game_design), ensure_ascii=False)
    return (
        "Implement the requested Minecraft mod features through the host-owned production "
        f"template. Request: {prompt}\nDesign context: {design}"
    )


def _implementation_research_outline(game_design: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "mod_id",
        "mod_name",
        "description",
        "features",
        "systems",
        "constraints",
        "acceptance_tests",
        "modules",
        "assets",
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
    brief = research_brief or central_research.normalize_research_brief(prompt, game_design)
    try:
        value = central_research.retrieve_domain_evidence(brief)
    except (SpecValidationError, ValueError, TypeError, RuntimeError):
        return {
            "schema_version": "mmm/research-unavailable-v1",
            "domains": [],
            "status": "unavailable",
        }
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "schema_version": "mmm/research-unavailable-v1",
        "domains": [],
        "status": "unavailable",
    }


def _generate_json_page(
    router: Any,
    *,
    system_prompt: str,
    request: Mapping[str, Any] | str,
    media_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Generate once and return a mapping; malformed output becomes an empty fill."""
    request_text = (
        request if isinstance(request, str) else json.dumps(request, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request_text},
    ]
    kwargs = {"media_paths": media_paths, "response_format": "json"}
    try:
        text = router.generate_text(
            "planner",
            messages,
            enable_tools=False,
            **kwargs,
        )
    except TypeError:
        try:
            text = router.generate_text("planner", messages, **kwargs)
        except (ValueError, RuntimeError):
            return {}
    except (ValueError, RuntimeError):
        return {}
    return _extract_json(str(text))


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the last JSON object without treating model shape as authority."""
    if not isinstance(text, str) or not text.strip():
        return {}
    objects = _json_objects(text)
    return dict(objects[-1]) if objects else {}


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


def _identifier(value: Any, fallback: str) -> str:
    import re

    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        text = re.sub(r"[^a-z0-9_]+", "_", fallback.lower()).strip("_") or "feature"
    if not text[0].isalpha():
        text = f"feature_{text}"
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
