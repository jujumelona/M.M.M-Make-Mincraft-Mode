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
from .model_router import ModelRouter
from .planner_template_schema import build_batch_skeleton
from .planning_pipeline import PlanningPipeline, PlanningStage, PlanningStageError


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
    """Canonical fail-closed complete planner.

    The live path is a compiler-like sequence owned by :class:`PlanningPipeline`:
    semantic design -> platform receipt -> target evidence -> evidence PlanIR ->
    production DAG. Runtime installers do not decide whether a failed stage may advance.
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
        artifacts = PlanningPipeline(self.router).prepare(
            prompt,
            media_paths=media_paths,
        )
        internal_design = {
            **artifacts.game_design,
            "_research_brief": artifacts.research_brief,
            "_technical_evidence": artifacts.technical_evidence,
        }

        try:
            evidence_plan = compile_evidence_first_plan(prompt, internal_design)
        except Exception as exc:
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                "evidence-first PlanIR compilation failed",
                cause=exc,
            ) from exc
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
        compiled = production_contract.compile_production_contract(
            requested_prompt=prompt,
            game_design=contract_design,
            research_brief=artifacts.research_brief,
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
        proposal = complete_proposal_from_parts(
            requested_prompt=prompt,
            base_proposal=artifacts.base_proposal,
            game_design=internal_design,
            modules=modules,
            assets=assets,
            acceptance_tests=tuple(compiled.acceptance_tests),
            existing_input_sha256=existing_input_sha256,
        )
        # Bind each verified reuse decision to its exact production owner before
        # live-target lowering. Without this handoff the planner can discover and
        # verify a donor while generation receives only the semantic task and is
        # forced to reimplement the capability from scratch.
        from .resource_asset_production import bind_reuse_plan

        proposal = bind_reuse_plan(proposal)
        from .live_module_lowering import lower_live_modules

        return lower_live_modules(self, proposal)

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
                raise PlanningStageError(
                    PlanningStage.EVIDENCE,
                    f"production batch {batch.batch_id!r} produced no valid modules",
                )

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
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                "production DAG contains no modules and no verified retain-only evidence",
            )

        tests = list(dict.fromkeys(test for test in tests if test))
        if not tests and modules:
            tests = [f"test_{module.module_id}_registers" for module in modules]
        return tuple(modules), tuple(assets), tuple(tests)


def _host_batches(prompt: str, game_design: Mapping[str, Any]) -> tuple[_ProductionBatch, ...]:
    """Compatibility helper for stored callers without evidence PlanIR."""
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

    if not exports:
        raise PlanningStageError(
            PlanningStage.DESIGN,
            "stored design has no explicit production modules",
        )
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


def _evidence_host_batches(plan: Mapping[str, Any]) -> tuple[_ProductionBatch, ...]:
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
                acceptance_tests=(),
            )
        )
    return tuple(batches)


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
    """Compatibility entrypoint. It is intentionally fail-closed."""
    existing = game_design.get("_platform_evidence")
    if isinstance(existing, Mapping):
        payload = dict(existing)
        if payload.get("status") == "unavailable":
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                "bound platform evidence is unavailable",
            )
        return payload
    brief = research_brief or central_research.normalize_research_brief(prompt, game_design)
    value = central_research.retrieve_domain_evidence(brief)
    if not isinstance(value, Mapping):
        raise PlanningStageError(
            PlanningStage.EVIDENCE,
            "central research returned a non-object evidence receipt",
        )
    payload = dict(value)
    if payload.get("status") == "unavailable":
        raise PlanningStageError(
            PlanningStage.EVIDENCE,
            "central research marked evidence unavailable",
        )
    return payload


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
