from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import production_contract
from .complete_spec import (
    AssetRequest,
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .evidence_execution_contract import task_batches
from .evidence_first_planning import compile_evidence_first_plan
from .model_router import ModelRouter
from .planner_template_schema import build_batch_skeleton
from .planning_pipeline import PlanningPipeline, PlanningStage, PlanningStageError
from .planner_trace_artifacts import repository_revision
from .research_derived_requirements import (
    attach_derived_requirement_ledger,
    derive_research_requirements,
)
from .root_cause_trace import emit_root_cause, trace_scope


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
    """Compile one authored request into a coder-ready production proposal.

    All mandatory plan structure is host-owned. The planner model is not asked to fill
    implementation holes, invent module identities, emit planning JSON, or decide whether
    the plan exists.
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
        from contextlib import nullcontext

        with trace_scope("complete_planning", trace_id=uuid.uuid4().hex):
            emit_root_cause(
                "planner_run_start",
                stage="planning",
                result="START",
                details=repository_revision(),
            )
            session_factory = getattr(self.router, "generation_session", None)
            session = session_factory("planner") if callable(session_factory) else nullcontext()
            with session:
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
                "host PlanIR compiler produced an invalid internal state",
                cause=exc,
            ) from exc

        # Materializing host batches here checks the deterministic DAG before any
        # optional enrichment is attached.
        _evidence_host_batches(evidence_plan)

        derived_ledger = derive_research_requirements(
            self.router,
            prompt=prompt,
            evidence_plan=evidence_plan,
            research_brief=artifacts.research_brief,
            technical_evidence=artifacts.technical_evidence,
            game_design=internal_design,
        )
        evidence_plan = attach_derived_requirement_ledger(evidence_plan, derived_ledger)

        internal_design = {
            **internal_design,
            "_derived_requirement_ledger": derived_ledger,
            "_evidence_first_plan": evidence_plan,
        }
        batches = _evidence_host_batches(evidence_plan)
        modules, assets, acceptance_tests = self._expand_batches(
            batches,
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
        facts_data, jobs_data = _lower_implementation_facts_and_jobs(
            modules, artifacts.base_proposal.spec
        )
        internal_design = {
            **internal_design,
            "production_outline": [_batch_dict(batch) for batch in batches],
            "_production_contract": compiled.contract,
            "_implementation_facts": facts_data,
            "_artifact_jobs": jobs_data,
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

        from .resource_asset_production import bind_reuse_plan

        proposal = bind_reuse_plan(proposal)
        from .live_module_lowering import lower_live_modules

        return lower_live_modules(self, proposal)

    def _expand_batches(
        self,
        batches: Sequence[_ProductionBatch],
        *,
        evidence_mode: bool = False,
        evidence_acceptance_tests: Sequence[str] = (),
    ) -> tuple[tuple[ProductionModule, ...], tuple[AssetRequest, ...], tuple[str, ...]]:
        """Lower the host PlanIR DAG directly into production modules.

        ``build_batch_skeleton`` already contains the evidence task, implementation
        template, dependencies, gates, acceptance and deliverables. There is therefore no
        model hole-fill stage and no model-output failure surface here.
        """
        modules: list[ProductionModule] = []
        assets: list[AssetRequest] = []
        tests: list[str] = list(dict.fromkeys(evidence_acceptance_tests))
        known_module_ids: set[str] = set()
        exports_by_batch: dict[str, tuple[str, ...]] = {}
        pending_batches = list(batches)
        completed_batch_ids: set[str] = set()

        while pending_batches:
            ready_batches = [
                batch
                for batch in pending_batches
                if set(batch.depends_on_batches).issubset(completed_batch_ids)
            ]
            if not ready_batches:
                # This can only be a host compiler bug because PlanIR is validated before
                # lowering. Preserve diagnostics instead of invoking another planner.
                unresolved = {
                    batch.batch_id: tuple(
                        dependency
                        for dependency in batch.depends_on_batches
                        if dependency not in completed_batch_ids
                    )
                    for batch in pending_batches
                }
                raise PlanningStageError(
                    PlanningStage.EVIDENCE,
                    f"host production dependency graph is internally inconsistent: {unresolved}",
                )

            for batch in ready_batches:
                dependency_ids = tuple(
                    module_id
                    for dependency in batch.depends_on_batches
                    for module_id in exports_by_batch.get(dependency, ())
                )
                page = build_batch_skeleton(
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
                expected_ids = {
                    str(item["module_id"])
                    for item in page["modules"]
                    if isinstance(item, Mapping) and item.get("module_id")
                }
                accepted_modules = [
                    _module(raw)
                    for raw in page["modules"]
                    if isinstance(raw, Mapping)
                    and str(raw.get("module_id") or "") in expected_ids
                    and str(raw.get("module_id") or "") not in known_module_ids
                ]
                if not accepted_modules:
                    # The skeleton builder always owns at least one export. Reaching this
                    # means host code is corrupt, not that another model turn is needed.
                    raise PlanningStageError(
                        PlanningStage.EVIDENCE,
                        f"host batch {batch.batch_id!r} lost all compiled module exports",
                    )

                modules.extend(accepted_modules)
                known_module_ids.update(module.module_id for module in accepted_modules)

                known_asset_ids = {item.asset_id for item in assets}
                known_asset_paths = {item.target_path for item in assets}
                for raw in page["assets"]:
                    if not isinstance(raw, Mapping):
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
                completed_batch_ids.add(batch.batch_id)
                pending_batches.remove(batch)

        if not modules and not evidence_mode:
            return (), tuple(assets), tuple(dict.fromkeys(tests))
        tests = list(dict.fromkeys(test for test in tests if test))
        if not tests and modules:
            tests = [f"test_{module.module_id}_registers" for module in modules]
        return tuple(modules), tuple(assets), tuple(tests)


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
            "derived_requirements": list(task.get("derived_requirements") or ()),
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


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            item.strip() for item in value if isinstance(item, str) and item.strip()
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


def _lower_implementation_facts_and_jobs(
    modules: Sequence[ProductionModule],
    spec: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .artifact_expansion import expand_facts_to_jobs
    from .implementation_fact import FactProvenance, ImplementationFact
    from .prompt_fact_types import FactType

    item_module_ids = {m.module_id for m in modules if m.kind == "item"}
    implementation_facts: list[ImplementationFact] = []
    for module in modules:
        if module.kind == "item":
            config = module.config if isinstance(module.config, dict) else {}
            display_name = str(
                config.get("name")
                or config.get("display_name")
                or config.get("display_name_en")
                or module.module_id.replace("_", " ").title()
            )
            implementation_facts.append(
                ImplementationFact(
                    fact_id=f"{module.module_id}.item_exists",
                    fact_type=FactType.ITEM_EXISTS,
                    subject=module.module_id,
                    display_name=display_name,
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=module.module_id,
                )
            )
            stack_limit = (
                config.get("stack_limit")
                or config.get("max_stack")
                or config.get("max_count")
            )
            if stack_limit is not None:
                try:
                    stack_val = int(stack_limit)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid stack limit for {module.module_id}: {stack_limit}"
                    ) from exc
                if not (1 <= stack_val <= 64):
                    raise ValueError(
                        f"Stack limit {stack_val} for {module.module_id} out of bounds [1, 64]"
                    )
                implementation_facts.append(
                    ImplementationFact(
                        fact_id=f"{module.module_id}.stack_limit",
                        fact_type=FactType.ITEM_STACK_LIMIT,
                        subject=module.module_id,
                        value=stack_val,
                        display_name=display_name,
                        provenance=FactProvenance.DESIGN,
                        parent_requirement=module.module_id,
                    )
                )
        elif module.kind == "block":
            config = module.config if isinstance(module.config, dict) else {}
            display_name = str(
                config.get("name")
                or config.get("display_name")
                or config.get("display_name_en")
                or module.module_id.replace("_", " ").title()
            )
            implementation_facts.append(
                ImplementationFact(
                    fact_id=f"{module.module_id}.block_exists",
                    fact_type=FactType.BLOCK_EXISTS,
                    subject=module.module_id,
                    display_name=display_name,
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=module.module_id,
                )
            )
            drop_item = str(
                config.get("drop")
                or config.get("drops")
                or config.get("drop_item")
                or config.get("loot_table_drop")
                or module.module_id
            )
            implementation_facts.append(
                ImplementationFact(
                    fact_id=f"{module.module_id}.block_drop",
                    fact_type=FactType.BLOCK_DROP,
                    subject=module.module_id,
                    object=drop_item,
                    display_name=display_name,
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=module.module_id,
                )
            )
            if drop_item not in item_module_ids and not any(
                f.fact_type == FactType.ITEM_EXISTS and f.subject == drop_item
                for f in implementation_facts
            ):
                implementation_facts.append(
                    ImplementationFact(
                        fact_id=f"{drop_item}.item_exists",
                        fact_type=FactType.ITEM_EXISTS,
                        subject=drop_item,
                        display_name=display_name,
                        provenance=FactProvenance.DESIGN,
                        parent_requirement=module.module_id,
                    )
                )

    artifact_jobs: list[dict[str, Any]] = []
    if implementation_facts:
        jobs = expand_facts_to_jobs(
            implementation_facts,
            mod_id=spec.mod_id,
            package_name=spec.package_name,
            main_class=getattr(spec, "main_class", "") or "",
        )
        artifact_jobs = [job.to_dict() for job in jobs]

    return [fact.to_dict() for fact in implementation_facts], artifact_jobs


__all__ = ["CompleteGameDesignPlanner"]

