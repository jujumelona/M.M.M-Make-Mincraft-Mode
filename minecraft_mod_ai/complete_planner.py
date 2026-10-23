from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from .platform_backend_contract import deterministic_backend_capabilities
from pathlib import Path
from typing import Any

from .authored_content_contract import content_owned_refs, content_request_catalog
from .authored_plan import AuthoredPlan
from .complete_spec import CompleteProposal
from .model_router import ModelRouter
from .planner_trace_artifacts import repository_revision
from .planner_checkpoint import PlannerCheckpoint
from .root_cause_trace import emit_root_cause, trace_scope


def _serialize_content_design(value: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the canonical content graph without leaking runtime dataclass objects."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "modules":
            result[key] = [
                {
                    "module_id": module.module_id,
                    "kind": module.kind,
                    "config": deepcopy(module.config),
                    "depends_on": list(module.depends_on),
                    "required_gates": list(module.required_gates),
                }
                for module in item
            ]
        elif key == "assets":
            result[key] = [asdict(asset) for asset in item]
        elif key == "_implementation_facts":
            result[key] = [fact.to_dict() for fact in item]
        else:
            result[key] = deepcopy(item)
    return result


class CompleteGameDesignPlanner:
    """Write a design freely; compile executable contracts only for production."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        adapter: Any = None,
        deterministic_module_kinds: Sequence[str] | frozenset[str] | None = None,
    ) -> None:
        self.router = router
        self.adapter = adapter
        self.deterministic_module_kinds = (
            frozenset(str(k).strip() for k in deterministic_module_kinds if str(k).strip())
            if deterministic_module_kinds is not None
            else None
        )

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
        adapter: Any = None,
        deterministic_module_kinds: Sequence[str] | frozenset[str] | None = None,
    ) -> AuthoredPlan:
        """Write the design itself; no schema, critic, evidence or production gate."""
        from .planner_operation import planner_operation

        from .authored_structured_design import (
            author_structured_sections,
            normalize_structured_sections,
            render_structured_sections,
        )
        from .planner_budget import PlannerBudget

        # Every planner loop is host-bounded at its own structural boundary.
        # Keep one ledger for accounting, but do not impose a second unrelated
        # global call ceiling across later typed-plan stages.
        budget = PlannerBudget()
        target_adapter = adapter or self.adapter or getattr(
            self.router, "_mmm_target_adapter", None,
        )
        target_version = str(
            getattr(target_adapter, "minecraft_version", "") or ""
        ).strip()
        target_loader = str(
            getattr(target_adapter, "loader", "") or ""
        ).strip()
        try:
            role_config = self.router.registry.role(self.router.profile, "planner")
            planner_model_id = str(getattr(role_config, "model_id", "") or "")
        except (AttributeError, KeyError, ValueError, TypeError):
            planner_model_id = str(getattr(self.router, "profile", "") or "")
        checkpoint = PlannerCheckpoint(
            prompt=prompt,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
            model_id=planner_model_id,
            target_version=target_version,
            target_loader=target_loader,
            source_revision=str(repository_revision().get("git_commit", "")),
        )
        from .request_scope import author_request_scope, request_scope_context

        request_scope = checkpoint.load_request_scope()
        if request_scope is None:
            request_scope = author_request_scope(self.router, prompt, budget=budget)
            checkpoint.save_request_scope(request_scope)
        planning_prompt = prompt + request_scope_context(request_scope)
        structured_sections = checkpoint.load_sections()
        if structured_sections is not None:
            try:
                structured_sections = normalize_structured_sections(structured_sections)
                print("PLAN_CHECKPOINT: restored structured design", flush=True)
            except ValueError:
                # Never resume from a stale or invalid authored design record.
                checkpoint.clear_sections()
                structured_sections = None
        restored_sections = structured_sections is not None
        if structured_sections is None:
            with planner_operation("author_structured_execution_contract"):
                structured_sections = author_structured_sections(
                    self.router,
                    planning_prompt,
                    media_paths=media_paths,
                    budget=budget,
                )
            structured_sections = normalize_structured_sections(structured_sections)
            checkpoint.save_sections(structured_sections)

        print("PLANNER_STAGE_COST: " + json.dumps({
            "stage": "author_design",
            "model_calls": budget.call_count,
            "breakdown": dict(sorted(budget.stage_counts.items())),
            "restored_checkpoint": restored_sections,
        }, ensure_ascii=False), flush=True)

        text = request_scope_context(request_scope).strip() + "\n\n" + render_structured_sections(structured_sections)
        return AuthoredPlan(
            requested_prompt=prompt,
            text=text,
            existing_input_sha256=existing_input_sha256,
            media_paths=tuple(str(path) for path in media_paths),
            structured_sections=structured_sections,
            content_design={},
            request_scope=request_scope,
        )

    def _author_for_production(self, plan: AuthoredPlan) -> AuthoredPlan:
        """Use the saved design to author executable code directly.

        Do not gate creation of the user-facing plan on availability of a
        specific state, currency, gameplay capability or Java backend.
        """
        from .planner_operation import planner_operation
        from .planner_budget import PlannerBudget
        from .authored_structured_design import (
            render_structured_sections,
        )
        from .typed_host_capabilities import typed_host_capability_contracts
        from .typed_plan_ir import validate_typed_host_capability_contracts
        from .typed_plan_authoring import author_typed_plan_ir

        prompt = plan.requested_prompt
        media_paths = plan.media_paths
        existing_input_sha256 = plan.existing_input_sha256
        structured_sections = plan.structured_sections
        from .request_scope import request_scope_context

        request_scope = plan.request_scope
        prior_content_design = dict(plan.content_design or {})
        for stale in ("_host_source_reuse", "_pre_design_source_handoff",
                      "_requested_curseforge_dependencies", "repository_candidates"):
            prior_content_design.pop(stale, None)
        adapter = self.adapter
        deterministic_module_kinds = self.deterministic_module_kinds
        budget = PlannerBudget()
        capability_contracts = typed_host_capability_contracts()
        validate_typed_host_capability_contracts(capability_contracts)
        target_adapter = adapter or getattr(self.router, "_mmm_target_adapter", None)
        from .typed_action_authoring import promote_action_sections
        promoted = promote_action_sections(self.router, structured_sections, budget=budget,
                                          minecraft_version=getattr(target_adapter, "minecraft_version", ""))
        if promoted != structured_sections:
            # UI ownership must refer to concrete, stable action identities.
            prior_content_design.pop("_request_runtime_owners", None)
        structured_sections = promoted
        # Reject an unexecutable typed UI action before building the content
        # graph or spending hundreds of model calls on a nonexistent mutation.
        # Never replace a presentation-only screen with a fabricated command.
        from .typed_action_authoring import preflight_ui_action_effects
        preflight_ui_action_effects(structured_sections)
        target_adapter = adapter or getattr(
            self.router, "_mmm_target_adapter", None,
        )
        target_version = str(
            getattr(target_adapter, "minecraft_version", "") or ""
        ).strip()
        target_loader = str(
            getattr(target_adapter, "loader", "") or ""
        ).strip()
        try:
            role_config = self.router.registry.role(self.router.profile, "planner")
            planner_model_id = str(getattr(role_config, "model_id", "") or "")
        except (AttributeError, KeyError, ValueError, TypeError):
            planner_model_id = str(getattr(self.router, "profile", "") or "")
        checkpoint = PlannerCheckpoint(
            prompt=prompt,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
            model_id=planner_model_id,
            target_version=target_version,
            target_loader=target_loader,
            source_revision=str(repository_revision().get("git_commit", "")),
        )

        # Production consumes the authored plan without injecting new features,
        # commands, resources or gameplay mechanics into the user's design.
        text = request_scope_context(request_scope) + render_structured_sections(structured_sections)

        content_catalog = content_request_catalog(
            structured_sections,
            requested_prompt=prompt,
            request_scope=request_scope,
        )
        content_design: dict[str, Any] = {}
        if content_catalog["requirements"]:
            from .content_design_graph import compile_content_graph

            content_mod_id = (
                "authored_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            )
            cached_design = checkpoint.load_content_design(content_catalog)
            if cached_design is not None:
                content_design = cached_design
                print("PLAN_CHECKPOINT: restored completed content graph", flush=True)
            else:
                restored_progress = checkpoint.content_progress(content_catalog)
                if restored_progress:
                    print(
                        "PLAN_CHECKPOINT: resumed content graph records="
                        f"{len(restored_progress)}",
                        flush=True,
                    )
                with planner_operation("author_content_design_graph"):
                    content_design = _serialize_content_design(
                        compile_content_graph(
                            prompt,
                            self.router,
                            request_catalog=content_catalog,
                            mod_id=content_mod_id,
                            progress=restored_progress,
                            checkpoint=checkpoint.save_content_record,
                        )
                    )
                checkpoint.save_content_design(content_catalog, content_design)
        # A resumed production plan may already carry host-stamped research
        # results. Preserve those receipts across content graph regeneration;
        # they remain reference-only until source inspection and compile proof.
        from copy import deepcopy

        for field in (
            "_planning_state",
        ):
            prior = prior_content_design.get(field)
            if isinstance(prior, Mapping):
                content_design[field] = deepcopy(dict(prior))
        external_content_refs = content_owned_refs(
            structured_sections,
            content_design,
        )
        from .request_runtime_ownership import author_runtime_ownership

        if request_scope and "_request_runtime_owners" not in content_design:
            content_design["_request_runtime_owners"] = author_runtime_ownership(
                self.router, request_scope, structured_sections, budget=budget,
            )
            checkpoint.save_content_design(content_catalog, content_design)
        runtime_owners = content_design.get("_request_runtime_owners", ())

        kinds = deterministic_module_kinds
        if kinds is None and adapter is not None:
            kinds = deterministic_backend_capabilities(adapter)
        if kinds is None:
            kinds = self.deterministic_module_kinds
        if kinds is None and self.adapter is not None:
            kinds = deterministic_backend_capabilities(self.adapter)
        if kinds is None:
            kinds = getattr(self.router, "_mmm_deterministic_module_kinds", None)
        if kinds is None:
            router_adapter = getattr(self.router, "_mmm_target_adapter", None)
            if router_adapter is not None:
                kinds = deterministic_backend_capabilities(router_adapter)
        if kinds is None:
            version = getattr(self.router, "_mmm_requested_minecraft_version", None)
            loader = getattr(self.router, "_mmm_requested_loader", None)
            if version and loader:
                from .platform_catalog import adapter_for_target

                resolved_adapter = adapter_for_target(
                    str(version),
                    str(loader),
                )
                kinds = deterministic_backend_capabilities(resolved_adapter)
        if kinds is None:
            existing_version = getattr(
                self.router,
                "_mmm_existing_minecraft_version",
                None,
            )
            existing_loader = getattr(
                self.router,
                "_mmm_existing_loader",
                None,
            )
            if existing_version and existing_loader:
                from .platform_catalog import adapter_for_target

                existing_adapter = adapter_for_target(
                    str(existing_version),
                    str(existing_loader),
                )
                kinds = deterministic_backend_capabilities(existing_adapter)

        # When no target is bound yet, do not pre-filter semantic platform kinds
        # using speculative provider discovery. The canonical target binder later
        # receives the authored module kinds and selects one immutable target via
        # platform_resolver._require_supported_kinds(). Duplicating that admission
        # here caused an empty/partial discovery frontier to collapse AUTO planning
        # to host-only kinds before the real target selection could run.
        auto_allowed_platform_kinds: frozenset[str] | None = None
        effective_kinds = tuple(sorted(kinds)) if kinds is not None else None

        with planner_operation("author_typed_plan_ir"):
            typed_plan_ir = author_typed_plan_ir(
                self.router,
                text,
                structured_sections,
                capability_contracts,
                deterministic_module_kinds=effective_kinds,
                allowed_platform_kinds=auto_allowed_platform_kinds,
                externally_covered_refs=external_content_refs,
                runtime_owners=runtime_owners,
                budget=budget,
            )
        from .typed_plan_support import assert_typed_plan_host_support

        assert_typed_plan_host_support(
            structured_sections,
            typed_plan_ir,
            externally_covered_refs=external_content_refs,
            runtime_owners=runtime_owners,
        )
        print("PLANNER_STAGE_COST: " + json.dumps({
            "stage": "author_for_production",
            "model_calls": budget.call_count,
            "breakdown": dict(sorted(budget.stage_counts.items())),
            "owned_runtime_groups": len(runtime_owners),
        }, ensure_ascii=False), flush=True)
        return AuthoredPlan(
            requested_prompt=prompt,
            text=text,
            existing_input_sha256=existing_input_sha256,
            media_paths=tuple(str(path) for path in media_paths),
            structured_sections=structured_sections,
            content_design=content_design,
            typed_plan_ir=typed_plan_ir,
            request_scope=request_scope,
        )

    def compile_for_production(
        self,
        prompt: AuthoredPlan,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> CompleteProposal:
        from .authored_production import compile_authored_design

        if not isinstance(prompt, AuthoredPlan):
            raise TypeError("compile_for_production requires AuthoredPlan")
        plan = prompt
        if not plan.typed_plan_ir:
            plan = self._author_for_production(plan)
        with trace_scope("production_preparation", trace_id=uuid.uuid4().hex):
            emit_root_cause(
                "production_preparation_start", stage="production", result="START",
                details={**repository_revision(), "input": "saved_authored_design"},
            )
            proposal = compile_authored_design(
                self.router, plan, existing_input_sha256=existing_input_sha256,
            )
            if proposal.assets:
                from .resource_asset_production import attach_generation_plan

                proposal = attach_generation_plan(self.router, proposal)
            return proposal

__all__ = ["CompleteGameDesignPlanner"]
