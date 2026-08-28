from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import central_research, production_contract
from .complete_spec import (
    AssetRequest,
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .evidence_first_planning import compile_evidence_first_plan, task_batches
from .game_design import GameDesignPlanner
from .model_router import ModelRouter
from .planner_template_schema import build_batch_skeleton
from .spec import SpecValidationError


@dataclass(frozen=True)
class _ProductionBatch:
    batch_id: str
    scope: str
    depends_on_batches: tuple[str, ...]
    deliverables: tuple[str, ...]
    exports: tuple[str, ...]
    task_contract: Mapping[str, Any] | None = None
    evidence_plan_sha256: str = ""
    acceptance_tests: tuple[str, ...] = ()


class CompleteGameDesignPlanner:
    """Plan production through deterministic host-owned batch templates.

    The model never creates the batch graph or schema. Host code derives the complete
    production template from the validated game design and creates every module identity.
    Evidence-first production batches are materialized directly from those host-owned
    templates; they do not ask the model to emit or repair JSON planning pages.
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

        evidence_plan = compile_evidence_first_plan(prompt, internal_design)
        internal_design = {
            **internal_design,
            "_evidence_first_plan": evidence_plan,
        }
        batches = _evidence_host_batches(evidence_plan)
        modules, assets, acceptance_tests = self._expand_batches(
            batches,
            prompt=prompt,
            game_design=internal_design,
            evidence_mode=True,
            evidence_acceptance_tests=tuple(
                str(check)
                for binding in evidence_plan["acceptance_release_bindings"]
                if isinstance(binding, Mapping)
                for check in binding.get("acceptance", ())
                if str(check).strip()
            ),
        )

        contract_design = {
            key: value
            for key, value in internal_design.items()
            if not str(key).startswith("_")
        }
        # Resolve through the module at call time so late runtime-finalization wrappers
        # (notably production_boundary_contract) are not bypassed by a stale imported
        # function alias captured before finalization completed.
        compiled = production_contract.compile_production_contract(
            requested_prompt=prompt,
            game_design=contract_design,
            research_brief=research_brief,
            modules=modules,
            assets=assets,
            acceptance_tests=acceptance_tests,
            evidence_plan=evidence_plan,
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
        evidence_mode: bool = False,
        evidence_acceptance_tests: Sequence[str] = (),
    ) -> tuple[tuple[ProductionModule, ...], tuple[AssetRequest, ...], tuple[str, ...]]:
        del prompt, game_design
        modules: list[ProductionModule] = []
        assets: list[AssetRequest] = []
        tests: list[str] = list(dict.fromkeys(evidence_acceptance_tests))
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
                host_module_contracts=(
                    {
                        module_id: {
                            **dict(batch.task_contract or {}),
                            "evidence_plan_sha256": batch.evidence_plan_sha256,
                            "evidence_task": dict(batch.task_contract or {}),
                        }
                        for module_id in batch.exports
                    }
                    if batch.task_contract is not None
                    else None
                ),
                acceptance_tests=batch.acceptance_tests,
            )
            # The host already owns the complete task graph, identities, dependencies,
            # completion predicates, and acceptance projection. Do not round-trip this
            # deterministic plan through an LLM JSON page: that only adds structured
            # recovery latency and lets internal execution language leak outward.
            page = skeleton

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

        if not modules and not evidence_mode:
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
    """Legacy compatibility helper for callers without an evidence-first contract.

    Live complete planning uses :func:`_evidence_host_batches`; this function remains
    available for stored callers and tests that construct the old minimal design shape.
    """
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


def _evidence_host_batches(plan: Mapping[str, Any]) -> tuple[_ProductionBatch, ...]:
    """Compile the validated semantic task DAG into host-owned production batches."""
    raw_batches = task_batches(plan)
    requirements = {
        str(item.get("requirement_id") or ""): dict(item)
        for item in plan.get("request_catalog", {}).get("requirements", [])
        if isinstance(item, Mapping) and item.get("requirement_id")
    }
    batches: list[_ProductionBatch] = []
    for raw in raw_batches:
        task = dict(raw["task_contract"])
        task["request_context"] = {
            "prompt_sha256": plan["request_catalog"]["prompt_sha256"],
            "requirements": [
                requirements[reference]
                for reference in task.get("requirement_refs", ())
                if reference in requirements
            ],
        }
        batches.append(
            _ProductionBatch(
                batch_id=str(raw["batch_id"]),
                scope=str(raw["scope"]),
                depends_on_batches=tuple(str(item) for item in raw["depends_on_batches"]),
                deliverables=tuple(str(item) for item in raw["deliverables"]),
                exports=tuple(str(item) for item in raw["exports"]),
                task_contract=task,
                evidence_plan_sha256=str(plan["plan_sha256"]),
                # Task-local integrity checks stay inside task_contract. Public/release
                # acceptance comes only from acceptance_release_bindings above.
                acceptance_tests=(),
            )
        )
    return tuple(batches)


def _implementation_research_outline(game_design: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded design outline expected by runtime platform contracts.

    This is a deterministic compatibility helper only. It performs no model call and
    does not reintroduce structured/JSON planner generation.
    """
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
    outline = {key: game_design[key] for key in keys if key in game_design}
    if (
        "_technical_evidence" in outline
        and "_platform_evidence" in outline
        and outline["_technical_evidence"] == outline["_platform_evidence"]
    ):
        outline.pop("_technical_evidence")
    return outline


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
    value = {
        "batch_id": batch.batch_id,
        "scope": batch.scope,
        "depends_on_batches": list(batch.depends_on_batches),
        "deliverables": list(batch.deliverables),
        "exports": list(batch.exports),
    }
    if batch.task_contract is not None:
        value["task_id"] = batch.batch_id
        value["evidence_plan_sha256"] = batch.evidence_plan_sha256
    return value


__all__ = ["CompleteGameDesignPlanner"]
